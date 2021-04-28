# Voice_Gernerative_Model_Embedding

## how to run
```
pip install -r requirements.txt

# trainimg models
## change below part in main.py
if __name__ == "__main__" : 

   # vae
   (vae_t, encoder_t, decoder_t,  x, y, x_train, x_test, y_test, lable_color_dict, group) = init_vae('RES')

   # train_vae (vae_t, encoder_t, decoder_t,  x, y, x_train, x_test, y_test, lable_color_dict, group, additional=False)

   # train_vae(vae_t, encoder_t, decoder_t,  x, y, x_train, x_test, y_test, lable_color_dict, group, True)

   train_classifier(vae_t, encoder_t, decoder_t,  x, y, x_train, x_test, y_test, lable_color_dict, group, True)

   test_classifier(vae_t, encoder_t, decoder_t,  x, y, x_train, x_test, y_test, lable_color_dict, group, False)
   
python main.py


# preprocessing and get result
## change below part in utils.py
if __name__ == "__main__" :
    # covert2mfcc('VCTK')
    # covert2mfcc('IMOECAP')
    # covert2mfcc('RES')
    # save_mfcc_np('RES')
    # save_mfcc_np('IMOECAP')
    classifier_result_intepretation('RES')

python utils.py

```
