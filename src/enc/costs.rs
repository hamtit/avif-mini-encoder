//! Symbol costs for mode information, derived from the current CDFs.

use crate::cdf::{CdfContext, costs_from_cdf};
use crate::consts::*;
use crate::tables::sizes::EXT_TX_IND;

pub struct ModeCosts {
    pub kf_y: [[[u32; 13]; 5]; 5],
    pub uv: [[[u32; 14]; 13]; 2],
    pub angle: [[u32; 7]; 8],
    pub fi_use: [[u32; 2]; 22],
    pub fi_mode: [u32; 5],
    pub cfl_sign: [u32; 8],
    pub cfl_alpha: [[u32; 16]; 6],
    pub skip: [[u32; 2]; 3],
    /// [tx size category][ctx][depth]
    pub tx_size: [[[u32; 3]; 3]; 4],
    /// [intra set index][square tx size][intra dir][tx type]
    pub tx_type: [[[[u32; 16]; 13]; 4]; 3],
    pub partition: [[u32; 10]; 20],
}

impl ModeCosts {
    pub fn new(c: &CdfContext) -> Box<ModeCosts> {
        let mut m = Box::new(ModeCosts {
            kf_y: [[[0; 13]; 5]; 5],
            uv: [[[0; 14]; 13]; 2],
            angle: [[0; 7]; 8],
            fi_use: [[0; 2]; 22],
            fi_mode: [0; 5],
            cfl_sign: [0; 8],
            cfl_alpha: [[0; 16]; 6],
            skip: [[0; 2]; 3],
            tx_size: [[[0; 3]; 3]; 4],
            tx_type: [[[[0; 16]; 13]; 4]; 3],
            partition: [[0; 10]; 20],
        });
        m.update(c);
        m
    }

    pub fn update(&mut self, c: &CdfContext) {
        for a in 0..5 {
            for l in 0..5 {
                costs_from_cdf(&c.kf_y[a][l], 13, &mut self.kf_y[a][l]);
            }
        }
        for y in 0..13 {
            costs_from_cdf(&c.uv_mode[0][y], 13, &mut self.uv[0][y]);
            costs_from_cdf(&c.uv_mode[1][y], 14, &mut self.uv[1][y]);
        }
        for i in 0..8 {
            costs_from_cdf(&c.angle_delta[i], 7, &mut self.angle[i]);
        }
        for b in 0..22 {
            costs_from_cdf(&c.filter_intra[b], 2, &mut self.fi_use[b]);
        }
        costs_from_cdf(&c.filter_intra_mode, 5, &mut self.fi_mode);
        costs_from_cdf(&c.cfl_sign, 8, &mut self.cfl_sign);
        for i in 0..6 {
            costs_from_cdf(&c.cfl_alpha[i], 16, &mut self.cfl_alpha[i]);
        }
        for i in 0..3 {
            costs_from_cdf(&c.skip[i], 2, &mut self.skip[i]);
        }
        for cat in 0..4 {
            let n = if cat == 0 { 2 } else { 3 };
            for ctx in 0..3 {
                costs_from_cdf(&c.tx_size[cat][ctx], n, &mut self.tx_size[cat][ctx]);
            }
        }
        let mut tmp = [0u32; 16];
        for (set_idx, set) in [(1usize, EXT_TX_SET_DTT4_IDTX_1DDCT), (2, EXT_TX_SET_DTT4_IDTX)] {
            let n = NUM_EXT_TX_SET[set];
            for sz in 0..4 {
                for dir in 0..13 {
                    costs_from_cdf(&c.intra_ext_tx[set_idx][sz][dir], n, &mut tmp);
                    for t in 0..16 {
                        if tx_in_set(set, t as u8) {
                            self.tx_type[set_idx][sz][dir][t] = tmp[EXT_TX_IND[set][t] as usize];
                        }
                    }
                }
            }
        }
        for ctx in 0..20 {
            let n = if ctx < 4 { 4 } else { 10 };
            costs_from_cdf(&c.partition[ctx], n, &mut self.partition[ctx]);
        }
    }

    /// Cost of the luma transform type (0 when it is not signalled).
    #[inline]
    pub fn tx_type_cost(&self, tx: TxSize, intra_dir: u8, t: TxType) -> u32 {
        let set = intra_tx_set(tx, false);
        if NUM_EXT_TX_SET[set] <= 1 {
            return 0;
        }
        self.tx_type[intra_tx_set_index(set)][tx.sqr() as usize][intra_dir as usize][t as usize]
    }
}
