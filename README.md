# dexcom_alert_detection
Trigger word detection of Dexcom alerts for Type 1 Diabetes blood glucose levels using TensorFlow.

To run this code you must download the ESC-50 dataset from 
[ESC](https://github.com/karolpiczak/ESC-50#download) and place the downloaded
folder into the data directory.

If you are having difficulty reading in the audio files with pydub like I was, try following the steps at this link (they helped me a lot!) [StackOverflow](https://stackoverflow.com/questions/77110765/error-while-run-command-ffmpeg-library-not-loaded-opt-homebrew-opt-mbedtls-l).