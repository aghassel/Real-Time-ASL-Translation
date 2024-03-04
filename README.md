4th Year Capstone Project Feature for AI Smart Glasses

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
