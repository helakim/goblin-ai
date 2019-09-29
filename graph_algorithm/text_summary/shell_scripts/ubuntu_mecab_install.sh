# *********************************************************************
# @Project    goblin-ai
# @FILE       kyung_tae_kim (hela.kim)
# @Copyright: hela
# @Created:   29/09/2019
#
#            7''  Q..\
#         _7         (_
#       _7  _/    _q.  /
#     _7 . ___  /VVvv-'_                                            .
#    7/ / /~- \_\\      '-._     .-'                      /       //
#   ./ ( /-~-/||'=.__  '::. '-~'' {             ___   /  //     ./{
#  V   V-~-~| ||   __''_   ':::.   ''~-~.___.-'' _/  // / {_   /  {  /
#   VV/-~-~-|/ \ .'__'. '.    '::                     _ _ _        ''.
#   / /~~~~||VVV/ /  \ )  \        _ __ ___   ___ ___(_) | | __ _   .::'
#  / (~-~-~\\.-' /    \'   \::::. | '_ ` _ \ / _ \_  / | | |/ _` | :::'
# /..\    /..\__/      '     '::: | | | | | | (_) / /| | | | (_| | ::
# vVVv    vVVv                 ': |_| |_| |_|\___/___|_|_|_|\__,_| ''
#
# *********************************************************************
echo
echo " __ __ ___   ___ ___(_) | | __ _      "
echo "| |_  | _ \ / _ \_  / | | |/ _| |     "
echo "| | | | | |  (_) / /| | | | (_| |     "
echo "|_| |_| |_| \___/___|_|_|_|\__,_| by kyung_tae_kim"
echo
echo "Press [Enter] to continue."
echo
read

CORPUS_DIR='cerberus_corpus'

if [ -d $CORPUS_DIR ]; then
  echo "step1. $CORPUS_DIR directory exists"
  echo "step2. $CORPUS_DIR directory deleted"
  rmdir $CORPUS_DIR
fi

if [ ! -e $CORPUS_DIR ]; then
  echo "---------------------------------------"
  echo "step1. $CORPUS_DIR directory not exists"
  echo "step2. $CORPUS_DIR directory crated"
  mkdir $CORPUS_DIR
fi

cd $CORPUS_DIR
curl -LO https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.1.tar.gz
tar zxfv mecab-0.996-ko-0.9.1.tar.gz
cd mecab-0.996-ko-0.9.1
./configure
make
make check
sudo make install

cd $CORPUS_DIR
curl -LO http://ftpmirror.gnu.org/automake/automake-1.11.tar.gz
tar -zxvf automake-1.11.tar.gz
cd automake-1.11
./configure
make
sudo make install

cd $CORPUS_DIR
curl -LO https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.0.1-20150920.tar.gz
tar -zxvf mecab-ko-dic-2.0.1-20150920.tar.gz
cd mecab-ko-dic-2.0.1-20150920
./autogen.sh
./configure
make
sudo sh -c 'echo "dicdir=/usr/local/lib/mecab/dic/mecab-ko-dic" > /usr/local/etc/mecabrc'
sudo make install

cd $CORPUS_DIR
git clone https://bitbucket.org/eunjeon/mecab-python-0.996.git
cd mecab-python-0.996

python setup.py build
python setup.py install

if hash "python3" &>/dev/null; then
  su
  python setup.py build
  python setup.py install
fi

pip install JPype1-py3