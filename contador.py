import tiktoken

def split_text_into_chunks(input_file, output_folder, chunk_size=1000):
    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()

    encoding_name = "cl100k_base"
    encoding = tiktoken.get_encoding(encoding_name)

    tokens = encoding.encode(text)
    total_tokens = len(tokens)

    num_chunks = (total_tokens // chunk_size) + (1 if total_tokens % chunk_size != 0 else 0)

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_tokens)

        chunk_tokens = tokens[start_idx:end_idx]
        chunk_text = encoding.decode(chunk_tokens)

        output_file = f"{output_folder}/chunk_{i + 1}.txt"

        with open(output_file, 'w', encoding='utf-8') as out_file:
            out_file.write(chunk_text)

    return num_chunks

# Example usage
input_txt_file = 'chaparro.txt'
output_folder = 'chunks_output'
chunk_size = 1000

num_chunks = split_text_into_chunks(input_txt_file, output_folder, chunk_size)

print(f"El archivo se dividió en {num_chunks} chunks. Archivos guardados en la carpeta '{output_folder}'.")
