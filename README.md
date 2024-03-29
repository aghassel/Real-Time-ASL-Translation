# 4th Year Capstone Project Feature for AI Smart Glasses
This Deep Learning project is a feature for AI Smart Glasses that aims to translate American Sign Language (ASL) in real time. It is part of a 4th Year Capstone Project. 

The dataset and model were provided by the Google Isolated Sign Language Recognition Challenge. The model uses an ensemble of four models composed of a 1DCNN combined with a Transformer to recognize and interpret ASL signs.

The images below show some of the most common words in the ASL dataset, the distribution of keypoint data, and a sample gesture for the "Shh" hand ASL signal.

> Note: For best results, use both hands. 

![Top-50](/images/ASL_dataset_top_50.png)

*Top 50 words from ASL-Signs Dataset*

![Keypoints](/images/keypoint_data_distribution.png)

*Keypoint Distribution*

![Sample Shh Gesture](/images/Shh-Hand.png)

*Sample gesture for Shh hand ASL signal*

**Installation**:

```python
pip3 install -r requirements.txt

python3 asl.py
```

Note for Raspberry Pi, modify the following instead of importing tensorflow:

```python
import tflite_runtime
    except:
        !pip install tflite-runtime

    import tflite_runtime.interpreter as tflite 
```

