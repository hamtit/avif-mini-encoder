use std::fmt;

/// Encoding errors.
///
/// The `Display` text is meant for logs and messages; match on the variant to
/// react to a specific failure.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// A side is zero or larger than 65536.
    InvalidDimensions { width: usize, height: usize },
    /// The frame would need more than one AV1 tile, which is not implemented.
    MultiTileUnsupported { width: usize, height: usize },
    /// The number of planes does not match the chroma format.
    PlaneCount { expected: usize, actual: usize },
    /// A plane buffer does not hold exactly `width * height` samples (subsampled for chroma).
    PlaneSize { plane: usize, expected: usize, actual: usize },
    /// The identity matrix (sRGB) is only valid for 4:4:4.
    IdentityRequires444,
    /// The thread encoding the alpha plane panicked.
    AlphaEncoderPanicked,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidDimensions { width, height } => write!(f, "unsupported size {width}x{height}"),
            Error::MultiTileUnsupported { width, height } => {
                write!(f, "{width}x{height} needs more than one tile, which is not implemented")
            }
            Error::PlaneCount { expected, actual } => write!(f, "expected {expected} plane(s), got {actual}"),
            Error::PlaneSize { plane, expected, actual } => {
                write!(f, "plane {plane} has {actual} bytes, expected {expected}")
            }
            Error::IdentityRequires444 => write!(f, "identity matrix (sRGB) requires 4:4:4"),
            Error::AlphaEncoderPanicked => write!(f, "alpha encoder panicked"),
        }
    }
}

impl std::error::Error for Error {}
