FROM ann-benchmarks

RUN apt-get install -y python-setuptools python-pip
RUN pip3 install numpy setuptools
RUN git clone https://github.com/dingodb/dingo-deploy.git;cd pydingo

RUN pip3 install -r requirements.txt
RUN pip3 uninstall dingodb
RUN python3 setup.py install

RUN python3 -c 'import dingodb'
