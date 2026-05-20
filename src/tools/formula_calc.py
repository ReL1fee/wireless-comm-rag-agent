"""
通信公式计算工具 — 计算常用通信公式
"""
import math
from pydantic import BaseModel, Field
from langchain_core.tools import tool


class FormulaInput(BaseModel):
    formula_name: str = Field(
        description="公式名称，可选: shannon_capacity, nyquist_rate, ber_bpsk, wavelength, path_loss"
    )
    bandwidth: float = Field(
        default=0, description="带宽，单位 Hz（用于 shannon_capacity, nyquist_rate）"
    )
    snr_db: float = Field(
        default=0, description="信噪比，单位 dB（用于 shannon_capacity）"
    )
    eb_n0_db: float = Field(
        default=0, description="Eb/N0，单位 dB（用于 ber_bpsk）"
    )
    frequency_hz: float = Field(
        default=0, description="频率，单位 Hz（用于 wavelength, path_loss）"
    )
    distance_m: float = Field(
        default=0, description="距离，单位 m（用于 path_loss）"
    )


@tool(args_schema=FormulaInput)
def calculate_formula(
    formula_name: str = "",
    bandwidth: float = 0,
    snr_db: float = 0,
    eb_n0_db: float = 0,
    frequency_hz: float = 0,
    distance_m: float = 0,
) -> str:
    """计算常用通信公式。当你需要计算香农容量、信噪比、误码率、
    奈奎斯特速率、波长、自由空间路径损耗时使用此工具。"""
    formula_name = formula_name.lower().strip()

    try:
        if formula_name in ("shannon_capacity", "shannon", "channel_capacity"):
            snr_linear = 10 ** (snr_db / 10)
            capacity = bandwidth * math.log2(1 + snr_linear)
            return (
                f"香农容量计算结果：\n"
                f"  带宽 B  = {bandwidth:.2e} Hz\n"
                f"  信噪比 = {snr_db:.2f} dB ({snr_linear:.4f} 线性)\n"
                f"  信道容量 C = {capacity:.2e} bps ({capacity/1e6:.2f} Mbps)"
            )

        elif formula_name in ("nyquist_rate", "nyquist"):
            rate = 2 * bandwidth
            return (
                f"奈奎斯特最大符号速率：\n"
                f"  带宽 B = {bandwidth:.2e} Hz\n"
                f"  最大符号速率 = {rate:.2e} Baud"
            )

        elif formula_name in ("ber_bpsk", "bpsk_ber", "ber"):
            eb_n0_linear = 10 ** (eb_n0_db / 10)
            ber = 0.5 * math.erfc(math.sqrt(eb_n0_linear))
            return (
                f"AWGN下BPSK理论误码率：\n"
                f"  Eb/N0 = {eb_n0_db:.2f} dB\n"
                f"  BER = {ber:.4e}"
            )

        elif formula_name in ("wavelength", "lambda"):
            c = 3e8
            wavelength = c / frequency_hz
            return (
                f"电磁波波长：\n"
                f"  频率 f = {frequency_hz:.2e} Hz\n"
                f"  波长 = {wavelength:.4f} m ({wavelength*100:.2f} cm)"
            )

        elif formula_name in ("path_loss", "friis", "free_space_path_loss"):
            c = 3e8
            wavelength = c / frequency_hz
            pl_db = 20 * math.log10(4 * math.pi * distance_m / wavelength)
            return (
                f"自由空间路径损耗（Friis公式）：\n"
                f"  距离 d = {distance_m:.2f} m\n"
                f"  频率 f = {frequency_hz:.2e} Hz\n"
                f"  路径损耗 = {pl_db:.2f} dB"
            )

        else:
            available = "shannon_capacity, nyquist_rate, ber_bpsk, wavelength, path_loss"
            return f"不支持的公式 '{formula_name}'。当前支持的公式：{available}"

    except (ValueError, TypeError, ZeroDivisionError) as e:
        return f"计算参数错误: {e}"
