import numpy as np
import pytest

from python.viz import uart


def _packets(seed=0, n=300):
    """A semantically valid device stream: contiguous SAMPLES, BEATs inside the sent range, STATS."""
    rng = np.random.default_rng(seed)
    out, idx = [], 0
    for _ in range(n):
        k = int(rng.integers(0, 10))
        if k < 7 or idx == 0:
            m = int(rng.integers(1, uart.MAX_SAMPLES_PER_FRAME + 1))
            out.append(uart.Samples(idx, tuple(int(v) for v in rng.integers(-32768, 32768, m)),
                                    tuple(int(v) for v in rng.integers(-2**31, 2**31, m))))
            idx += m
        elif k < 9:
            r = idx - 1 - int(rng.integers(0, min(idx, 200)))
            out.append(uart.Beat(r, idx - 1, int(rng.integers(0, 4)), int(rng.integers(0, 4)),
                                 tuple(int(v) for v in rng.integers(-2**31, 2**31, 4)), 300, 290, 2400, 8))
        else:
            out.append(uart.Stats(idx - 1, 5, 123456, 200000, 50_000_000, 200, 12, 9, 0x1234ABCD))
    return out


def _encode(p):
    if isinstance(p, uart.Samples):
        return uart.encode_samples(p.first_idx, p.adc, p.filt)
    if isinstance(p, uart.Beat):
        return uart.encode_beat(p)
    return uart.encode_stats(p)


def _decode(data: bytes, dec=None, chunk_seed=None) -> tuple[list, uart.FrameDecoder]:
    dec = dec if dec is not None else uart.HostDecoder()
    if chunk_seed is None:
        got = dec.feed(data)
    else:
        rng, got, k = np.random.default_rng(chunk_seed), [], 0
        while k < len(data):
            m = int(rng.integers(1, 40))
            got += dec.feed(data[k:k + m])
            k += m
    return got + dec.end(), dec


def _in_order_subsequence(sub, seq) -> bool:
    it = iter(seq)
    return all(any(q == g for q in it) for g in sub)


# ---------- CRC-16 (frames) and CRC-32 (beat digest) ----------

def test_crc16_check_value():
    # CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no reflection, xorout 0
    assert uart.crc16(b"123456789") == 0x29B1
    assert uart.crc16(b"") == 0xFFFF
    assert uart.crc16(b"A") == 0xB915


def _crc16_bitwise(data: bytes) -> int:
    c = 0xFFFF
    for b in data:
        c ^= b << 8
        for _ in range(8):
            c = ((c << 1) ^ 0x1021) & 0xFFFF if c & 0x8000 else (c << 1) & 0xFFFF
    return c


def test_crc16_table_equals_bitwise():
    rng = np.random.default_rng(7)
    for n in (1, 2, 5, 33, 101, 255, 257):
        d = rng.integers(0, 256, size=n).astype(np.uint8).tobytes()
        assert uart.crc16(d) == _crc16_bitwise(d)


def test_crc32_digest_check_value_and_chaining():
    import zlib
    assert uart.crc32_update(0, b"123456789") == 0xCBF43926 == zlib.crc32(b"123456789")
    assert uart.crc32_update(uart.crc32_update(0, b"1234"), b"56789") == 0xCBF43926
    beats = [p for p in _packets(seed=9) if isinstance(p, uart.Beat)]
    assert uart.beat_digest(beats) == zlib.crc32(b"".join(uart.beat_payload(b) for b in beats))
    assert uart.beat_digest([]) == 0


def test_frame_layout_and_crc():
    fr = uart.encode_stats(uart.Stats(1, 2, 3, 4, 5, 200, 12, 9, 0xCBF43926))
    assert fr[:2] == b"\xa5\x5a" and fr[2] == uart.T_STATS and fr[3] == uart.LEN_STATS == 28
    assert fr[-2:] == uart.crc16(fr[2:-2]).to_bytes(2, "little")   # CRC over type, len, payload; little-endian
    assert fr[-6:-2] == (0xCBF43926).to_bytes(4, "little")          # beat_digest = last STATS field
    assert len(uart.encode_beat(uart.Beat(0, 0, 0, 0, (0, 0, 0, 0), 0, 0, 0, 0))) == 6 + uart.LEN_BEAT == 39
    assert len(uart.encode_samples(0, [0] * 8, [0] * 8)) == 6 + 5 + 6 * 8 == 59


def test_encoder_rejects_bad_sample_count():
    with pytest.raises(ValueError):
        uart.encode_samples(0, [], [])
    with pytest.raises(ValueError):
        uart.encode_samples(0, [0] * 17, [0] * 17)


# ---------- clean stream ----------

@pytest.mark.parametrize("decoder", [uart.FrameDecoder, uart.HostDecoder])
def test_roundtrip_any_chunking(decoder):
    pk = _packets()
    got, dec = _decode(b"".join(_encode(p) for p in pk), decoder(), chunk_seed=1)
    assert got == pk
    assert dec.bad_header == dec.bad_crc == dec.transport_error == dec.skipped_bytes == 0


def test_frames_accepted_without_waiting_for_the_next_one():
    pk = _packets(n=5)
    dec = uart.HostDecoder()
    assert dec.feed(b"".join(_encode(p) for p in pk)) == pk          # no lookahead: the last frame too
    assert dec.end() == []


def test_split_sync_word_kept():
    s = uart.Samples(0, (1,), (2,))
    st = uart.Stats(1, 2, 3, 4, 5, 200, 12, 9, 7)
    fr = uart.encode_stats(st)
    dec = uart.HostDecoder()
    assert dec.feed(b"\x00\x13" + _encode(s) + fr[:1]) == [s]       # half a sync word stays buffered
    assert dec.feed(fr[1:]) == [st]


# ---------- host checks (a) ----------

def test_host_beat_checks():
    s0, s1 = uart.Samples(0, (0,) * 8, (0,) * 8), uart.Samples(8, (0,) * 8, (0,) * 8)
    ok_beat = uart.Beat(15, 15, 1, 0, (0, 0, 0, 0), 300, 300, 2400, 8)
    bad = [uart.Beat(16, 16, 1, 0, (0, 0, 0, 0), 300, 300, 2400, 8),      # R beyond the received samples
           uart.Beat(10, 16, 4, 0, (0, 0, 0, 0), 300, 300, 2400, 8)]      # label outside 0..3
    got, dec = _decode(b"".join(_encode(p) for p in [s0, s1, *bad, ok_beat]))
    assert got == [s0, s1, ok_beat]
    assert dec.transport_error == 2 and dec.samples_lost == 0 and dec.resyncs == 0


def test_host_accepts_discontinuous_samples_and_resyncs():
    a = uart.Samples(0, (0,) * 8, (0,) * 8)
    b = uart.Samples(8 + 100, (0,) * 8, (0,) * 8)                         # 100 samples lost
    c = uart.Samples(116 + 5000, (0,) * 8, (0,) * 8)                      # long outage: 5000 lost
    d = uart.Samples(3, (0,) * 8, (0,) * 8)                               # backwards (e.g. device reset)
    beat_old = uart.Beat(2, 20, 0, 0, (0, 0, 0, 0), 300, 300, 2400, 8)    # before the restarted range
    beat_new = uart.Beat(5, 20, 0, 0, (0, 0, 0, 0), 300, 300, 2400, 8)
    got, dec = _decode(b"".join(_encode(p) for p in [a, b, c, d, beat_old, beat_new]))
    assert got == [a, b, c, d, beat_new]                                  # every SAMPLES frame accepted
    assert dec.samples_lost == 100 + 5000 and dec.resyncs == 3 and dec.transport_error == 1
    assert (dec.start_idx, dec.next_idx) == (3, 11)


# ---------- corrupted streams ----------

def test_resync_after_garbage_between_frames():
    """Garbage (and fake sync words) between frames: the decoder resynchronises and recovers every frame;
    nothing false is accepted."""
    pk = _packets(seed=2)
    rng = np.random.default_rng(3)
    parts = []
    for p in pk:
        g = rng.integers(0, 256, size=int(rng.integers(0, 20))).astype(np.uint8).tobytes() if rng.random() < 0.3 else b""
        if rng.random() < 0.1:
            g += b"\xa5\x5a\x02"                     # fake sync + type, then the real frame follows
        parts.append(g + _encode(p))
    got, dec = _decode(b"".join(parts), chunk_seed=4)
    assert got == pk
    assert dec.transport_error == 0 and dec.skipped_bytes > 0


def _drop_one_byte_stream(seed: int, n_damaged: int = 25, garbage: bool = False):
    """25 frames lose one byte; with garbage=True 1..11 random bytes also follow each damaged frame
    (the same damage as live_monitor --corrupt)."""
    pk = _packets(seed=seed)
    rng = np.random.default_rng(seed)
    frames = [_encode(p) for p in pk]
    damaged = set(rng.choice(len(frames), size=n_damaged, replace=False).tolist())
    data = b""
    for i, fr in enumerate(frames):
        if i in damaged:
            b = bytearray(fr)
            del b[int(rng.integers(0, len(b)))]      # one byte lost on the line
            if garbage:
                b += rng.integers(0, 256, size=int(rng.integers(1, 12))).astype(np.uint8).tobytes()
            fr = bytes(b)
        data += fr
    return pk, damaged, data


@pytest.mark.parametrize("seed", [4, 5, 6])
def test_resync_after_dropped_bytes(seed):
    pk, damaged, data = _drop_one_byte_stream(seed)
    got, dec = _decode(data)
    assert all(g in pk for g in got)                                     # no false accept
    assert _in_order_subsequence(got, pk)
    # an intact frame can be lost only by the BEAT range check after a lost SAMPLES frame
    lost = [i for i, p in enumerate(pk) if i not in damaged and p not in got]
    assert all(isinstance(pk[i], uart.Beat) for i in lost)
    assert len(got) >= len(pk) - 2 * len(damaged)


def test_no_false_accepts_2500_dropped_byte_frames():
    """The measurement reported in docs/uart_protocol.md: seeds 4..103, 25 damaged frames each."""
    fa_host = 0
    for seed in range(4, 104):
        for garbage in (False, True):
            pk, damaged, data = _drop_one_byte_stream(seed, garbage=garbage)
            fa_host += sum(g not in pk for g in _decode(data, uart.HostDecoder())[0])
    assert fa_host == 0                  # CRC-16 + host checks: no false accept


def _one_of_each():
    return [uart.Samples(8, (1, -2, 300, -32768, 32767, 0, 5, 9), (7, -8, 2**31 - 1, -2**31, 0, 1, 2, 3)),
            uart.Samples(8, (100,), (-100,)),
            uart.Beat(5, 7, 2, 1, (-5, 2**31 - 1, -2**31, 77), 72, 1080, 8640, 8),
            uart.Stats(7, 1, 0, 0, 0, 200, 12, 9, 0xFFFFFFFF)]


@pytest.mark.parametrize("pkt", _one_of_each(), ids=["samples8", "samples1", "beat", "stats"])
def test_any_single_bit_flip_rejects_the_frame(pkt):
    """Every single-bit error anywhere in a frame (sync, type, len, payload, CRC) rejects that frame;
    the surrounding frames are still decoded."""
    head = uart.Samples(0, (0,) * 8, (0,) * 8)
    tail = uart.Samples(16, (0,) * 8, (0,) * 8)
    fr = _encode(pkt)
    assert _decode(_encode(head) + fr + _encode(tail))[0] == [head, pkt, tail]
    for bit in range(8 * len(fr)):
        b = bytearray(fr)
        b[bit // 8] ^= 1 << (bit % 8)
        got, dec = _decode(_encode(head) + bytes(b) + _encode(tail))
        assert pkt not in got, bit
        assert got == [head, tail], (bit, got)
