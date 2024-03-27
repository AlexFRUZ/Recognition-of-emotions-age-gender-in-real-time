def sub_bytes(state):
    s_box_string = '63 7c 77 7b f2 6b 6f c5 30 01 67 2b fe d7 ab 76' \
                   'ca 82 c9 7d fa 59 47 f0 ad d4 a2 af 9c a4 72 c0' \
                   'b7 fd 93 26 36 3f f7 cc 34 a5 e5 f1 71 d8 31 15' \
                   '04 c7 23 c3 18 96 05 9a 07 12 80 e2 eb 27 b2 75' \
                   '09 83 2c 1a 1b 6e 5a a0 52 3b d6 b3 29 e3 2f 84' \
                   '53 d1 00 ed 20 fc b1 5b 6a cb be 39 4a 4c 58 cf' \
                   'd0 ef aa fb 43 4d 33 85 45 f9 02 7f 50 3c 9f a8' \
                   '51 a3 40 8f 92 9d 38 f5 bc b6 da 21 10 ff f3 d2' \
                   'cd 0c 13 ec 5f 97 44 17 c4 a7 7e 3d 64 5d 19 73' \
                   '60 81 4f dc 22 2a 90 88 46 ee b8 14 de 5e 0b db' \
                   'e0 32 3a 0a 49 06 24 5c c2 d3 ac 62 91 95 e4 79' \
                   'e7 c8 37 6d 8d d5 4e a9 6c 56 f4 ea 65 7a ae 08' \
                   'ba 78 25 2e 1c a6 b4 c6 e8 dd 74 1f 4b bd 8b 8a' \
                   '70 3e b5 66 48 03 f6 0e 61 35 57 b9 86 c1 1d 9e' \
                   'e1 f8 98 11 69 d9 8e 94 9b 1e 87 e9 ce 55 28 df' \
                   '8c a1 89 0d bf e6 42 68 41 99 2d 0f b0 54 bb 16'
    s_box = bytearray.fromhex(s_box_string)

    for i in range(len(state)):
        for j in range(len(state[i])):
            state[i][j] = s_box[state[i][j]]
    return state


def shift_rows(state):
    for i in range(1, 4):
        state[i] = state[i][i:] + state[i][:i]
    return state


def mix_columns(state):
    for j in range(4):
        s0 = state[0][j]
        s1 = state[1][j]
        s2 = state[2][j]
        s3 = state[3][j]

        state[0][j] = multiply(s0, 0x02) ^ multiply(s1, 0x03) ^ s2 ^ s3
        state[1][j] = s0 ^ multiply(s1, 0x02) ^ multiply(s2, 0x03) ^ s3
        state[2][j] = s0 ^ s1 ^ multiply(s2, 0x02) ^ multiply(s3, 0x03)
        state[3][j] = multiply(s0, 0x03) ^ s1 ^ s2 ^ multiply(s3, 0x02)

        for i in range(4):
            state[i][j] %= 0x100

    return state


def multiply(a, b):
    result = 0
    for _ in range(8):
        if b & 1:
            result ^= a
        a <<= 1
        if a & 0x100:
            a ^= 0x11b
        b >>= 1
    return result


def test_aes():
    input_data = [
        [0x19, 0xa0, 0x9a, 0xe9],
        [0x3d, 0xf4, 0xc6, 0xf8],
        [0xe3, 0xe2, 0x8d, 0x48],
        [0xbe, 0x2b, 0x2a, 0x08]
    ]

    substituted_state = sub_bytes(input_data)
    print("Після заміни байтів:")
    print_state(substituted_state)

    shifted_state = shift_rows(substituted_state)
    print("\nПісля зсуву рядків:")
    print_state(shifted_state)

    mixed_state = mix_columns(shifted_state)
    print("\nПісля міксування колонок:")
    print_state(mixed_state)


def print_state(state):
    for row in state:
        print(' '.join(format(x, '02x') for x in row))


test_aes()
