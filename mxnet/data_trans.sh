# channerduan
# transform the data for mxnet
path='../mnist_raw/'
pwd
ls $path

rows=35000
if [ $# -ne 0 ];then
   rows=$1
else
	echo "using default train and valid sets split:" $rows
fi

sed -n "2,$rows p" $path'train.csv' | cut -c 3- >data.csv
sed -n "2,$rows p" $path'train.csv' | cut -c 1 >data_label.csv
sed -n "$[$rows+1],\$p" $path'train.csv' | cut -c 3- >valid.csv
sed -n "$[$rows+1],\$p" $path'train.csv' | cut -c 1 >valid_label.csv
sed -n '2,$p' $path'test.csv' > test_ready.csv
