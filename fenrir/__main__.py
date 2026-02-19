"""
Allow running FENRIR as a module: python -m fenrir
"""

import asyncio
from fenrir.bot import main

if __name__ == "__main__":
    asyncio.run(main())
