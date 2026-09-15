//! Memory module - chat.dat append-only log and chat.idx index
//! Binary format: [u64 ts][u8 role][u16 importance*100][u32 len][len bytes utf8]

use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdxEntry {
    pub offset: u64,
    pub len: u32,
    pub ts: u64,
    pub importance: f32,
    pub reward: f32,
    pub access_count: u32,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Role {
    User = 0,
    Assistant = 1,
    Tool = 2,
}

impl Role {
    fn from_u8(v: u8) -> Self {
        match v {
            0 => Role::User,
            1 => Role::Assistant,
            2 => Role::Tool,
            _ => Role::User,
        }
    }
}

pub struct DiskLog {
    log_file: File,
    idx_path: String,
    index: Vec<IdxEntry>,
}

impl DiskLog {
    pub fn new(log_path: &str, idx_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let log_file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(log_path)?;
        
        let index = Self::load_index(idx_path)?;
        
        Ok(Self {
            log_file,
            idx_path: idx_path.to_string(),
            index,
        })
    }
    
    fn load_index(path: &str) -> Result<Vec<IdxEntry>, Box<dyn std::error::Error>> {
        if !Path::new(path).exists() {
            return Ok(Vec::new());
        }
        
        let mut file = File::open(path)?;
        let mut entries = Vec::new();
        
        // Each entry is 32 bytes: u64 + u32 + u64 + f32 + f32 + u32
        const ENTRY_SIZE: usize = 32;
        let mut buf = [0u8; ENTRY_SIZE];
        
        loop {
            match file.read_exact(&mut buf) {
                Ok(_) => {
                    let offset = u64::from_le_bytes(buf[0..8].try_into().unwrap());
                    let len = u32::from_le_bytes(buf[8..12].try_into().unwrap());
                    let ts = u64::from_le_bytes(buf[12..20].try_into().unwrap());
                    let importance = f32::from_le_bytes(buf[20..24].try_into().unwrap());
                    let reward = f32::from_le_bytes(buf[24..28].try_into().unwrap());
                    let access_count = u32::from_le_bytes(buf[28..32].try_into().unwrap());
                    
                    entries.push(IdxEntry {
                        offset,
                        len,
                        ts,
                        importance,
                        reward,
                        access_count,
                    });
                }
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }
        }
        
        Ok(entries)
    }
    
    pub fn save_index(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = File::create(&self.idx_path)?;
        const ENTRY_SIZE: usize = 32;
        let mut buf = [0u8; ENTRY_SIZE];
        
        for entry in &self.index {
            buf[0..8].copy_from_slice(&entry.offset.to_le_bytes());
            buf[8..12].copy_from_slice(&entry.len.to_le_bytes());
            buf[12..20].copy_from_slice(&entry.ts.to_le_bytes());
            buf[20..24].copy_from_slice(&entry.importance.to_le_bytes());
            buf[24..28].copy_from_slice(&entry.reward.to_le_bytes());
            buf[28..32].copy_from_slice(&entry.access_count.to_le_bytes());
            file.write_all(&buf)?;
        }
        
        file.sync_all()?;
        Ok(())
    }
    
    pub fn append(&mut self, role: Role, importance: f32, text: &str) -> Result<u64, Box<dyn std::error::Error>> {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();
        
        let offset = self.log_file.seek(SeekFrom::End(0))?;
        
        // Write header: [u64 ts][u8 role][u16 importance*100][u32 len]
        let mut header = [0u8; 15];
        header[0..8].copy_from_slice(&ts.to_le_bytes());
        header[8] = role as u8;
        header[9..11].copy_from_slice(&((importance * 100.0) as u16).to_le_bytes());
        header[11..15].copy_from_slice(&(text.len() as u32).to_le_bytes());
        
        self.log_file.write_all(&header)?;
        self.log_file.write_all(text.as_bytes())?;
        self.log_file.sync_all()?;
        
        // Add to index
        let entry = IdxEntry {
            offset,
            len: (15 + text.len()) as u32,
            ts,
            importance,
            reward: 0.5,
            access_count: 0,
        };
        self.index.push(entry);
        
        Ok(offset)
    }
    
    pub fn read_at(&mut self, offset: u64) -> Result<(Role, f32, String), Box<dyn std::error::Error>> {
        self.log_file.seek(SeekFrom::Start(offset))?;
        
        let mut header = [0u8; 15];
        self.log_file.read_exact(&mut header)?;
        
        let ts = u64::from_le_bytes(header[0..8].try_into().unwrap());
        let role = Role::from_u8(header[8]);
        let importance = u16::from_le_bytes(header[9..11].try_into().unwrap()) as f32 / 100.0;
        let len = u32::from_le_bytes(header[11..15].try_into().unwrap()) as usize;
        
        let mut buf = vec![0u8; len];
        self.log_file.read_exact(&mut buf)?;
        
        let text = String::from_utf8_lossy(&buf).to_string();
        Ok((role, importance, text))
    }
    
    pub fn get_index(&self) -> &[IdxEntry] {
        &self.index
    }
    
    pub fn update_reward(&mut self, idx: usize, reward: f32) {
        if idx < self.index.len() {
            let entry = &mut self.index[idx];
            entry.reward = entry.reward * 0.95 + reward * 0.05;
        }
    }
    
    pub fn increment_access(&mut self, idx: usize) {
        if idx < self.index.len() {
            self.index[idx].access_count += 1;
        }
    }
}
