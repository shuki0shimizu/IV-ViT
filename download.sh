ROOT_FILE="pretrain_weight/"
mkdir $ROOT_FILE

FILE_ID="17_4XmIgx1lqz4SYBY3GQVwrE7R0nFFKj";
FILE_NAME="video_model_kinetics_pretrain.pth";
CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://drive.google.com/uc?export=download&id=$FILE_ID" -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1/p');
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$CONFIRM&id=$FILE_ID" -O $ROOT_FILE$FILE_NAME;
rm -f /tmp/cookies.txt

FILE_ID="1yfT73g6EWVfsk4w3ub7uS0LjtyhXvWOb";
FILE_NAME="video_model_imagenet21k_pretrain.pth";
CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://drive.google.com/uc?export=download&id=$FILE_ID" -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1/p');
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$CONFIRM&id=$FILE_ID" -O $ROOT_FILE$FILE_NAME;
rm -f /tmp/cookies.txt
