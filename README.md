<div align="center">
  
# Automatic Restoration of Diacritics for Speech Data Set
Sara Shatnawi, Sawsan Alqahtani, Hanan Aldarmaki <br> 
**NAACL 2024**

[![image](https://github.com/SaraShatnawi/Diacritization/assets/49264609/11fcd298-8569-417a-93f8-fbd3d6764dd2)](https://aclanthology.org/2024.naacl-long.233.pdf)

</div>

## Abstract
Automatic text-based diacritic restoration models generally have high diacritic error rates when applied to speech transcripts as a result of domain and style shifts in spoken language. In this work, we explore the possibility of improving the performance of automatic diacritic restoration when applied to speech data by utilizing parallel spoken utterances. In particular, we use the pre-trained Whisper ASR model fine-tuned on relatively small amounts of diacritized Arabic speech data to produce rough diacritized transcripts for the speech utterances, which we then use as an additional input for diacritic restoration models. The proposed framework consistently improves diacritic restoration performance compared to text-only baselines. Our results highlight the inadequacy of current text-based diacritic restoration models for speech data sets and provide a new baseline for speech-based diacritic restoration.

<img width="1318" alt="attention_example_2" src="https://github.com/SaraShatnawi/Diacritization/assets/49264609/09e84346-5682-49a0-aa57-4a2e5a34f7ef">

## Models
1. Text-based with Tashkeela: a text-only model trained on Tashkeela and fine-tuned with CLArTTS.
2. Text-based without Tashkeela: a text-only model trained only on CLArTTS.
3. Text+ASR with Tashkeela: a Text+ASR model trained on Tashkeela for text and fine-tuned with CLArTTS.
4. Text+ASR without Tashkeelh: a Text+ASR model trained only with CLArTTS.
* For each one of the above, there are Transformer and LSTM versions for the text encoders.

Text+ASR models use an external ASR system, a fine-tuned Whisper, to pre-process speech. You can [find the fine-tuned whisper here](huggingface.co/sashat/whisper-medium-ClassicalAr).

## ![image](https://github.com/SaraShatnawi/Diacritization/assets/49264609/19d1f469-f0fc-4346-9dc8-38c017dbd8fc) Environment & Installation
<h3> Prerequisites</h3>


* Tested with Python 3.8
* Install the required packages listed in requirements.txt file

   * pip install -r requirements.txt

 ## Citition
 If you use the above model, please cite the following paper: 
 ```
  @inproceedings{shatnawi2024automatic,
  title={Automatic Restoration of Diacritics for Speech Data Sets},
  author={Shatnawi, Sara and Alqahtani, Sawsan and Aldarmaki, Hanan},
  booktitle={Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
  pages={4166--4176},
  year={2024}
  }
```

<div align="center">

<br><br>


# Data Augmentation for Speech-Based Diacritic Restoration
Sara Shatnawi, Sawsan Alqahtani, Shady Shehata, Hanan Aldarmaki <br> 
Mohamed bin Zayed University of Artificial Intelligence <br>
**ArabicNLP 2024**

</div>

## Abstract
This paper describes a data augmentation technique for boosting the performance of speech-based diacritic restoration. Our experiments demonstrate the utility of this approach, resulting in improved generalization of all models across different test sets. In addition, we describe the first multi-modal diacritic restoration model, utilizing both speech and text as input modalities. This type of model can be used to diacritize speech transcripts. Unlike previous work that relies on an external ASR model, the proposed model is far more compact and efficient. While the multi-modal framework does not surpass the ASR-based model for this task, it offers a promising approach for improving the efficiency of speech-based diacritization, with a potential for improvement using data augmentation and other methods.

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

## Multi-Modal Diacritic Restoration
![image](https://github.com/SaraShatnawi/Data-Augmentation/assets/49264609/20652f27-3ebb-40d2-b77e-a3df7ec1a6c9)


 ## Citition
If you use the data augmentation or multi-modal model, please cite the following paper:
 ```
    @inproceedings{shatnawi2024data,
    title={Data Augmentation for Speech-Based Diacritic Restoration},
    author={Shatnawi, Sara and Alqahtani, Sawsan and Shehata, Shady and Aldarmaki, Hanan},    
    booktitle={Proceedings of ArabicNLP 2024},
    year={2024}
    }
```
    


