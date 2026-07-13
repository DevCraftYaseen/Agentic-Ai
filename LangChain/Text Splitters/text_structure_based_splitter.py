from langchain_text_splitters import RecursiveCharacterTextSplitter

text = '''
Space exploration has led to incredible seientific discoveries. From landing on the Moon to
exploring Mars, humanity continues to push the boundaries of whaths possible beyond our
planet.

These missions have not only expanded our knowledge of the universe but have also
contributed to advancements in technology here on Earth. Satellite communications, GPS, and
even certain medical imaging techniques trace their roots back to innovations driven by
space programs.
'''

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0,
)
chunks = text_splitter.split_text(text)

print(len(chunks))
print(chunks)

