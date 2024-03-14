def s_box(key):
    key_length = len(key)
    if key_length == 0:
        exit()
    S = list(range(256))
    j = 0
    for i in range(256):
        j = (j + S[i] + ord(key[i % key_length])) % 256
        S[i], S[j] = S[j], S[i]
    return S


def pseudo_random_generate_algorithm(S, n):
    i = 0
    j = 0
    key_stream = []
    for _ in range(n):
        i = (i + 1) % 256
        j = (j + S[i]) % 256
        S[i], S[j] = S[j], S[i]
        key_stream.append(S[(S[i] + S[j]) % 256])
    return key_stream


def RC4(key, text):
    S = s_box(key)
    key_stream = pseudo_random_generate_algorithm(S, len(text))
    cipher_text = [chr(ord(text[i]) ^ key_stream[i]) for i in range(len(text))]
    return ''.join(cipher_text)


while True:
    key = input("Enter a key: ")
    plaintext = input("Enter the text for the cipher: ")

    cipher_text = RC4(key, plaintext)
    print("Encrypted:", cipher_text)

    decrypted_text = RC4(key, cipher_text)
    print("Decrypted:", decrypted_text)