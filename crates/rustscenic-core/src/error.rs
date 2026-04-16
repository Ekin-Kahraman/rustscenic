use thiserror::Error;

#[derive(Error, Debug)]
pub enum ScenicError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("dimension mismatch: {0}")]
    Dimension(String),
}

pub type Result<T> = std::result::Result<T, ScenicError>;
