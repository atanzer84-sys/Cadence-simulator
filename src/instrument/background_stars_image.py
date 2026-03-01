import logging
import numpy as np
import astropy.units as u
from loaders.run_waltzer_context import RunContext
from configs.channel_config import SpectroscopyChannel
from domain.star import Star
import logging
import numpy as np
import math
from astropy.table import Table
from loaders.run_waltzer_context import get_repo_root

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import join
from astroquery.gaia import Gaia
from configs.global_config import GlobalConfig


def generate_Background_Stars_Image(channel: SpectroscopyChannel, ctx: RunContext, star: Star):
    pass
