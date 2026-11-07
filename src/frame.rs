//! Planar 8-bit image buffers.

#[derive(Clone)]
pub struct Plane {
    pub data: Vec<u8>,
    pub stride: usize,
    /// Visible size.
    pub width: usize,
    pub height: usize,
    /// Allocated (padded) size.
    pub alloc_w: usize,
    pub alloc_h: usize,
}

impl Plane {
    pub fn new(width: usize, height: usize, alloc_w: usize, alloc_h: usize) -> Plane {
        Plane { data: vec![0; alloc_w * alloc_h], stride: alloc_w, width, height, alloc_w, alloc_h }
    }

    /// Copy `src` (width x height, `src_stride`) and replicate the right and
    /// bottom edges into the padding.
    pub fn from_slice(
        src: &[u8],
        src_stride: usize,
        width: usize,
        height: usize,
        alloc_w: usize,
        alloc_h: usize,
    ) -> Plane {
        let mut p = Plane::new(width, height, alloc_w, alloc_h);
        for y in 0..alloc_h {
            let sy = y.min(height - 1);
            let row = &src[sy * src_stride..sy * src_stride + width];
            let dst = &mut p.data[y * alloc_w..(y + 1) * alloc_w];
            dst[..width].copy_from_slice(row);
            let last = row[width - 1];
            dst[width..].fill(last);
        }
        p
    }

    #[inline]
    pub fn at(&self, x: usize, y: usize) -> u8 {
        self.data[y * self.stride + x]
    }

    #[inline]
    pub fn offset(&self, x: usize, y: usize) -> usize {
        y * self.stride + x
    }
}
