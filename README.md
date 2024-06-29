# Towards Automatic Restoration of Diacritics for Speech Data Sets
Sara Shatnawi, Sawsan Alqahtani, Hanan Aldarmaki <br> 
Mohamed bin Zayed University of Artificial Intelligence, Abu Dhabi, UAE <br>
[![image](https://github.com/SaraShatnawi/Diacritization/assets/49264609/11fcd298-8569-417a-93f8-fbd3d6764dd2)](https://arxiv.org/abs/2311.10771)

# Introduction
We propose a diacritic restoration framework for speech data. Unlike traditional text-based methods, this model leverages the power of both speech and text. It utilizes a fine-tuned pre-trained automatic speech recognition model (Whisper) to generate an initial, noisy diacritized version of the speech transcript. This noisy transcript is then combined with the original text and fed into a diacritic restoration model.  The proposed approach achieved significant improvements in diacritic restoration performance compared to text-only models, paving the way for more robust speech-based diacritic restoration.

<img width="1318" alt="attention_example_2" src="https://github.com/SaraShatnawi/Diacritization/assets/49264609/09e84346-5682-49a0-aa57-4a2e5a34f7ef">

# Models
1. Text-based with Tashkeela: a text-only model trained on Tashkeela and fine-tuned with CLArTTS.
2. Text-based without Tashkeela: a text-only model trained only on CLArTTS.
3. Text+ASR with Tashkeela: a Text+ASR model trained on Tashkeela for text and fine-tuned with CLArTTS.
4. Text+ASR without Tashkeelh: a Text+ASR model trained only with CLArTTS.
* For each one of the above, there are Transformer and LSTM versions.

You can find the fine-tuned whisper here [here](huggingface.co/sashat/whisper-medium-ClassicalAr).

# ![image](https://github.com/SaraShatnawi/Diacritization/assets/49264609/19d1f469-f0fc-4346-9dc8-38c017dbd8fc) Environment & Installation
<h3> Prerequisites</h3>


* Tested with Python 3.8
* Install the required packages listed in requirements.txt file

   * pip install -r requirements.txt

 # Citition
    @inproceedings{shatnawi2023automatic,
    title={Automatic Restoration of Diacritics for Speech Data Sets},
    author={Shatnawi, Sara and Alqahtani, Sawsan and Aldarmaki, Hanan},    
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)”,
    year = "2024”,
    publisher = "Association for Computational Linguistics",
}

# Data Augmentation for Speech-Based Diacritic Restoration
Sara Shatnawi, Sawsan Alqahtani, Shady Shehata, Hanan Aldarmaki <br> 
Mohamed bin Zayed University of Artificial Intelligence, Abu Dhabi, UAE <br>
## Data Augmentation Rules 
### Replacement Rules
* Sukoon or Shaddah if they appear at the first letter of the word (i.e.,مْقعد ).
* Tanween if it appears in any letter except the last letter in the word.
* One of the two Shaddahs appearing on two contiguous characters (i.e.,مَقّعّد ).
* Any diacritic that is not Fatha or Damma appearing on Hamza on top (أ) at the beginning of a word (example of allowed variations, in this case, أَصبحَ or أُصبحَ ).
* Any diacritic that is not Kasra appearing on Hamza below Alef (إ), such as the word إلى.
* Any diacritic that is not Fatha before the tied T (ة) (i.e., the Arabic word مَدرَسَة).
* Any diacritic other than Fatha before the letter Alef of the following forms: ( ى ) or ( ا ).
* Stand-alone Shadda should be followed by another diacritic.

### Deletion Rules
* All diacritics are placed on characters, not in the Arabic alphabet.
* All diacritics applied to the following forms of Alef: Alef Madd (آ), Alef (ا), Maqsura (ى), and at the beginning of a word (Alef followed by the letter Lam) indicating the definiteness of a word (ال).
* Any additional diacritic for each letter, except if this additional diacritic accompanies Shaddah (i.e., each letter should have only one diacritic except in the case of Shaddah, which can be followed by an additional diacritic).

# Multi-Modal Diacritic Restoration
![image](https://github.com/SaraShatnawi/Data-Augmentation/assets/49264609/20652f27-3ebb-40d2-b77e-a3df7ec1a6c9)


 # Citition
    @inproceedings{shatnawi2024data,
    title={Data Augmentation for Speech-Based Diacritic Restoration},
    author={Shatnawi, Sara and Alqahtani, Sawsan and and Shehata, Shady Aldarmaki, Hanan},    
    booktitle = "SIGARAB ArabicNLP 2024 Conference”,
    year = "2024",
    }
    


