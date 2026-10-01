// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later

/// Storage geometry for a packed weight format accepted by
/// `gpu.tensor.linear`. Keeping this table shared prevents the checker,
/// interpreter, and generated host guards from disagreeing about a format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct QuantizedLinearGeometry {
    pub block_elements: usize,
    pub block_bytes: usize,
    pub scale_offset: usize,
}

pub(crate) const SUPPORTED_QUANTIZED_LINEAR_FORMATS: &str =
    "q8_0, q5_0, q4_0, iq4_nl, q6_k, q4_k, q3_k, q2_k";

pub(crate) fn quantized_linear_geometry(name: &str) -> Option<QuantizedLinearGeometry> {
    let geometry = match name {
        "q8_0" => QuantizedLinearGeometry { block_elements: 32, block_bytes: 34, scale_offset: 0 },
        "q5_0" => QuantizedLinearGeometry { block_elements: 32, block_bytes: 22, scale_offset: 0 },
        "q4_0" | "iq4_nl" => QuantizedLinearGeometry { block_elements: 32, block_bytes: 18, scale_offset: 0 },
        "q6_k" => QuantizedLinearGeometry { block_elements: 256, block_bytes: 210, scale_offset: 208 },
        "q4_k" => QuantizedLinearGeometry { block_elements: 256, block_bytes: 144, scale_offset: 0 },
        "q3_k" => QuantizedLinearGeometry { block_elements: 256, block_bytes: 110, scale_offset: 108 },
        "q2_k" => QuantizedLinearGeometry { block_elements: 256, block_bytes: 84, scale_offset: 80 },
        _ => return None,
    };
    Some(geometry)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_linear_geometry_covers_legacy_and_k_quant_formats() {
        assert_eq!(quantized_linear_geometry("q5_0").unwrap().block_bytes, 22);
        assert_eq!(quantized_linear_geometry("q3_k").unwrap().scale_offset, 108);
        assert_eq!(quantized_linear_geometry("q2_k").unwrap().block_elements, 256);
        assert_eq!(quantized_linear_geometry("unknown"), None);
    }
}
