//! GGUF weight loader - parses GGUF file format for tensor metadata
//! Does NOT use mmap - uses seek + read_exact only

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub offset: u64,
    pub size: u64,
    pub qtype: u32,
    pub layer: u16,
    pub is_expert: bool,
    pub expert_id: u8,
}

#[derive(Debug)]
pub struct GgufLoader {
    pub tensor_map: HashMap<String, TensorMeta>,
    pub file: File,
}

impl GgufLoader {
    pub fn new(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut file = File::open(path)?;
        let tensor_map = Self::parse_header(&mut file)?;
        Ok(Self { tensor_map, file })
    }
    
    fn parse_header(file: &mut File) -> Result<HashMap<String, TensorMeta>, Box<dyn std::error::Error>> {
        // Read GGUF magic
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if &magic != b"GGUF" {
            return Err("Invalid GGUF file".into());
        }
        
        // Read version (u32)
        let mut version_buf = [0u8; 4];
        file.read_exact(&mut version_buf)?;
        let _version = u32::from_le_bytes(version_buf);
        
        // Read tensor_count (u64) and kv_count (u64)
        let mut buf8 = [0u8; 8];
        file.read_exact(&mut buf8)?;
        let tensor_count = u64::from_le_bytes(buf8);
        file.read_exact(&mut buf8)?;
        let _kv_count = u64::from_le_bytes(buf8);
        
        // Skip KV section (we only need tensors)
        // For simplicity, read all key-value pairs and discard
        for _ in 0.._kv_count {
            Self::skip_string(file)?;
            Self::skip_value(file)?;
        }
        
        // Parse tensor infos
        let mut tensor_map = HashMap::new();
        for i in 0..tensor_count {
            let name = Self::read_string(file)?;
            
            // n_dims (u32)
            let mut buf4 = [0u8; 4];
            file.read_exact(&mut buf4)?;
            let _n_dims = u32::from_le_bytes(buf4);
            
            // shape (array of u64, length = n_dims)
            for _ in 0.._n_dims {
                file.read_exact(&mut buf8)?;
            }
            
            // type (u32) - quantization type
            file.read_exact(&mut buf4)?;
            let qtype = u32::from_le_bytes(buf4);
            
            // offset (u64)
            file.read_exact(&mut buf8)?;
            let offset = u64::from_le_bytes(buf8);
            
            // Determine layer from name (e.g., "blk.0.ffn_gate")
            let layer = Self::extract_layer(&name);
            let (is_expert, expert_id) = Self::extract_expert_info(&name);
            
            // Size will be computed later when we know the full layout
            tensor_map.insert(name, TensorMeta {
                offset,
                size: 0, // Will be filled in next pass
                qtype,
                layer,
                is_expert,
                expert_id,
            });
        }
        
        // Compute sizes by looking at consecutive offsets
        let data_start = file.stream_position()?;
        let mut offsets: Vec<(String, u64)> = tensor_map.iter()
            .map(|(k, v)| (k.clone(), v.offset))
            .collect();
        offsets.sort_by_key(|(_, o)| *o);
        
        for i in 0..offsets.len() {
            let (name, offset) = offsets[i].clone();
            let size = if i + 1 < offsets.len() {
                offsets[i + 1].1 - offset
            } else {
                // Last tensor - estimate from file size
                file.seek(SeekFrom::End(0))?;
                let file_len = file.stream_position()?;
                file_len - offset
            };
            
            if let Some(meta) = tensor_map.get_mut(&name) {
                meta.size = size;
            }
        }
        
        // Seek back to start of data for reading
        file.seek(SeekFrom::Start(data_start))?;
        
        Ok(tensor_map)
    }
    
    fn extract_layer(name: &str) -> u16 {
        // Parse "blk.N.xxx" format
        if let Some(start) = name.find("blk.") {
            let rest = &name[start + 4..];
            if let Some(end) = rest.find('.') {
                return rest[..end].parse().unwrap_or(0);
            }
        }
        0
    }
    
    fn extract_expert_info(name: &str) -> (bool, u8) {
        if name.contains("ffn_gate_exps") || name.contains("ffn_up_exps") || name.contains("ffn_down_exps") {
            // Extract expert ID from name like "blk.0.ffn_gate_exps.3"
            if let Some(pos) = name.rfind('.') {
                if let Ok(id) = name[pos + 1..].parse::<u8>() {
                    return (true, id);
                }
            }
            (true, 0)
        } else {
            (false, 0)
        }
    }
    
    fn read_string(file: &mut File) -> Result<String, Box<dyn std::error::Error>> {
        let len = Self::read_u64(file)?;
        let mut buf = vec![0u8; len as usize];
        file.read_exact(&mut buf)?;
        Ok(String::from_utf8_lossy(&buf).to_string())
    }
    
    fn skip_string(file: &mut File) -> Result<(), Box<dyn std::error::Error>> {
        let len = Self::read_u64(file)?;
        file.seek(SeekFrom::Current(len as i64))?;
        Ok(())
    }
    
    fn skip_value(file: &mut File) -> Result<(), Box<dyn std::error::Error>> {
        // Read type byte
        let mut type_byte = [0u8; 1];
        file.read_exact(&mut type_byte)?;
        let gguf_type = type_byte[0];
        
        // Skip based on type
        match gguf_type {
            0 => {} // UINT8 - 1 byte
            1 => { file.seek(SeekFrom::Current(1))?; } // INT8
            2 => { file.seek(SeekFrom::Current(2))?; } // INT16
            3 => { file.seek(SeekFrom::Current(4))?; } // INT32
            4 => { file.seek(SeekFrom::Current(8))?; } // INT64
            5 => { file.seek(SeekFrom::Current(4))?; } // FLOAT32
            6 => { file.seek(SeekFrom::Current(8))?; } // FLOAT64
            7 => { Self::skip_string(file)?; } // BOOL (stored as string in some versions)
            8 => { Self::skip_string(file)?; } // STRING
            9 => { // ARRAY
                let elem_type = Self::read_u8(file)?;
                let len = Self::read_u64(file)?;
                for _ in 0..len {
                    Self::skip_array_element(file, elem_type)?;
                }
            }
            _ => {}
        }
        Ok(())
    }
    
    fn skip_array_element(file: &mut File, elem_type: u8) -> Result<(), Box<dyn std::error::Error>> {
        match elem_type {
            0 | 1 => {}
            2 => { file.seek(SeekFrom::Current(2))?; }
            3 => { file.seek(SeekFrom::Current(4))?; }
            4 => { file.seek(SeekFrom::Current(8))?; }
            5 => { file.seek(SeekFrom::Current(4))?; }
            6 => { file.seek(SeekFrom::Current(8))?; }
            7 | 8 => { Self::skip_string(file)?; }
            _ => {}
        }
        Ok(())
    }
    
    fn read_u64(file: &mut File) -> Result<u64, Box<dyn std::error::Error>> {
        let mut buf = [0u8; 8];
        file.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }
    
    fn read_u8(file: &mut File) -> Result<u8, Box<dyn std::error::Error>> {
        let mut buf = [0u8; 1];
        file.read_exact(&mut buf)?;
        Ok(buf[0])
    }
    
    pub fn read_tensor(&mut self, name: &str, buf: &mut [u8]) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(meta) = self.tensor_map.get(name) {
            self.file.seek(SeekFrom::Start(meta.offset))?;
            self.file.read_exact(buf)?;
            Ok(())
        } else {
            Err(format!("Tensor {} not found", name).into())
        }
    }
    
    pub fn get_meta(&self, name: &str) -> Option<&TensorMeta> {
        self.tensor_map.get(name)
    }
}
