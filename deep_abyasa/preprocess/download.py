import os.path
import urllib.request
import tarfile


class Download:
    """Download dataset and supporting files from URL and unzips them. If the
    dataset already exists at the given path, it will not be downloaded again.

    Args:
        dataset: Dataset to download. Currently, available datasets are:

                1. chem_struct_to_elem

        root: root path. Default value is https://storage.googleapis.com/chem-dl

        save_at: Path to save the downloaded and extracted datasets/files.
            Default values is current directory.

    """
    def __init__(self, dataset, root="https://storage.googleapis.com/chem-dl", save_at="."):
        self.root = root

        self.dataset = dataset
        self.dataset_file = f'{dataset}.tar.gz'

        self.save_at = f'{save_at}'
        self.save_at_file = f'{save_at}/{self.dataset_file}'

        self.path = f'{root}/{self.dataset_file}'

        self.has_downloaded = False

        self.download_dataset()
        self.extract_dataset()

    def download_dataset(self):
        """Download dataset from given URL and save the contents to path specified
           at self.save_at. If the file already exists at self.save_at, no
           download will occur
        """
        if os.path.isfile(self.save_at_file):
            print(f'{self.save_at_file} exists. Nothing will be downloaded')
            return
        else:
            try:
                urllib.request.urlcleanup()
                urllib.request.urlretrieve(self.path, f'{self.save_at}/{self.dataset_file}')
                self.has_downloaded = True
            except Exception as e:
                print(f'Failed to download {self.dataset} from {self.root}: {e}')
                raise e

    def extract_dataset(self):
        """Extracts contents of tarred file to self.save_at. If the extracted
           directory already exists, no untarring will occur
        """
        if os.path.isdir(f'{self.save_at}/{self.dataset}') and self.has_downloaded is False:
            print(f'{self.save_at}/{self.dataset} exists. Nothing will be unzipped')
            return
        else:
            try:
                tar = tarfile.open(self.save_at_file, "r:gz")
                tar.extractall(self.save_at)
            except Exception as e:
                print(f'Failed to extract {self.save_at}/{self.dataset}')
                raise e
            finally:
                tar.close()


