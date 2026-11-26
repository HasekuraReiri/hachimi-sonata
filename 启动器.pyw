#!/usr/bin/env python
"""Windows 双击启动入口（.pyw 不显示控制台）。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from launcher import main

if __name__ == "__main__":
    sys.exit(main())
