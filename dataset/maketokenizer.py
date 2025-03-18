from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import glob

# TODO: Re-think Tokenizer-type, and parameters. Add prefix/suffix for word continuation.

if __name__ == "__main__":
	tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
	tokenizer.pre_tokenizer = Whitespace()
	files = glob.glob("./merged_chunk_*.txt")

	print("Found files:", *files, sep='\n')

	trainer = BpeTrainer(special_tokens=["[UNK]", "[MASK]", "[CLS]", "[PAD]", "[RVD]"], vocab_size=16384, show_progress=True, limit_alphabet=1024, max_token_length=12)

	print("Training tokenizer..")

	tokenizer.train(files, trainer)
	tokenizer.save("tokenizer-owtsubset05.json")
