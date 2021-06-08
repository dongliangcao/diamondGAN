from distutils.core import setup
setup(
  name = 'DiamondGAN',        
  packages = ['DiamondGAN'],  
  version = '0.2',      
  license='MIT',        
  description = 'DiamondGAN: GAN used for multimodal translation in medical imaging from MRI T1, T2 to MRI FLAIR, DIR',  
  author = 'Cao Dongliang',                  
  author_email = 'cao.dongliang97@gmail.com',   
  url = 'https://https://github.com/dongliangcao',   
  download_url = 'https://github.com/dongliangcao/diamondGAN/archive/refs/tags/v0.2-alpha.tar.gz',    
  keywords = ['CycleGAN', 'multimodal translation', 'MRI'],   
  install_requires=[          
          'numpy',
          'tensorflow',
          'tensorflow_addons',
          'keras',
          'SimpleITK'
      ],
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