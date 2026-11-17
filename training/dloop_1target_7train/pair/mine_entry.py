"""Start the native own-method trainer with the agreed shared batch schedule."""
from .shared_packing import install

install()
from four_anchor.cli import main

if __name__=='__main__': main()
