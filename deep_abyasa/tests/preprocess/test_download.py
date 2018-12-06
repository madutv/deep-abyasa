from deep_abyasa import Download
import os.path
import shutil

def test_download_unzip():
    Download('temp_2')
    assert(os.path.isfile('temp_2.tar.gz'))
    assert(os.path.isdir('temp_2'))
    os.remove('temp_2.tar.gz')
    shutil.rmtree('temp_2')