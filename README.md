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
        @misc{shatnawi2023automatic,                                                                            
      title={Automatic Restoration of Diacritics for Speech Data Sets}, 
      author={Sara Shatnawi and Sawsan Alqahtani and Hanan Aldarmaki},
      year={2023},
      eprint={2311.10771},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

