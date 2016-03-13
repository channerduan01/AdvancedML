# run in CentOS 7, CPU only
yum install -y vim
yum install -y git
yum -y groupinstall "Development Tools"
yum install -y python-devel

curl -O https://bootstrap.pypa.io/get-pip.py
python get-pip.py
pip install virtualenv

virtualenv venv
source ~/venv/bin/activate
pip install pandas -v
pip install --upgrade -v https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.7.1-cp27-none-linux_x86_64.whl

git clone https://github.com/channerduan01/AdvancedML
cd AdvancedML/tensorflow

# nohup python mnist.py &
python mnist.py
