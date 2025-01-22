https://github.com/yzyouzhang/AIR-ASVspoof 코드 참고  
Wav2Vec2_LCNN_ocsoftmax 보다 WavLM_MFA 폴더 코드에 ocsoftmax 관련 잘 정리되어 있습니다.   

### WavLM_MFA 폴더 ocsoftmax 설명
- data_utils_ocsoftmax.py : 기존 dev set 불러올 때 label을 안 불러왔는데 불러오도록 변경(그 외 수정사항 x)
- evaluation_ocsoftmax.py : eer 계산 코드 변경
- utils.py : Class ocsoftmax 추가
- main_ wavlm_base_mfaclassifier_ocsoftmax.py : 기존 model의 last_hidden값을 받아 ocsoftmax를 통과시켜 loss 계산 후 학습하도록 코드 수정(ocsoftmax, loss_model로 되어있으니 검색 후 해당 부분 참고)

