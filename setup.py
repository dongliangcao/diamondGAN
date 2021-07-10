import setuptools

with open("README.txt", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
  name = 'DiamondGAN',         
  version = '2.0',      
  license='MIT',        
  description = 'DiamondGAN: GAN used for multimodal translation in medical imaging from MRI T1, T2 to MRI FLAIR, DIR',  
  long_description=long_description,
  long_description_content_type="text/markdown",
  author = 'Cao Dongliang',                  
  author_email = 'cao.dongliang97@gmail.com',   
  url = 'https://github.com/dongliangcao/diamondGAN/',   
  download_url = 'https://github.com/dongliangcao/diamondGAN/archive/refs/tags/v2.0.tar.gz',    
  keywords = ['CycleGAN', 'multimodal translation', 'MRI'],   
  install_requires=[          
          'numpy',
          'tensorflow',
          'tensorflow_addons',
          'keras',
          'SimpleITK'
      ],
  package_dir={"": "DiamondGAN"},
  packages=setuptools.find_packages(where="DiamondGAN"),
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.8',
  ],
)