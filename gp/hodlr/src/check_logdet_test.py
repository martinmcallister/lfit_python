#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np

t = np.arange(50) / 24. / 60.
diag = np.ones_like(t)

a, s = 0.999, 5.0
matrix = a*a*np.exp(-0.5 * (t[:, None] - t[None, :]) / (s*s))
matrix[(range(len(t)), range(len(t)))] = diag

print(np.linalg.slogdet(matrix))
