"""(optional) visulalization functions for the hdrplus algorithm.
Copyright (c) 2021 Antoine Monod

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU Affero General Public License
as published by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program.
If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import cv2


def addMotionField(image, tileSize, motionVectors, overlap=True):
	'''draws motion vectors as arrows on top of the input image
	(arrows start from the center of each tile)
	Args:
		image: 2d array
		tileSize: uint, length of the square tiles
		motionVectors: 2D array of tuples (u, v) extracted from motion estimation
		overlap: bool, wether or not motionVectors relates to image tiles overlapped by half in each spatial dimension
	'''
	motionDrawing = image.copy()
	motionVectors = np.round(motionVectors).astype(int)

	params = {
		'color' : (130, 130, 255),
		'thickness' : 2,
		'tipLength' : .2
	}
	if overlap:
		c = 0
		colors = [(130, 130, 255), (255, 130, 130), (130, 255, 130), (130, 130, 130)]
		for di, dj in zip([0, 1, 0, 1], [0, 0, 1, 1]):
			params['color'] = colors[c]
			motionVectorsSub = motionVectors[di::2, dj::2]

			for i in range(motionVectorsSub.shape[0]):
				for j in range(motionVectorsSub.shape[1]):
					anchor = (int((j + dj / 2 + 1 / 2) * tileSize), int((i + di / 2 + 1 / 2) * tileSize))  # opencv points are described (x,y), ie (column, row)
					tip = (anchor[0] + motionVectorsSub[i, j, 1], anchor[1] + motionVectorsSub[i, j, 0])
					motionDrawing = cv2.arrowedLine(motionDrawing, anchor, tip, **params)
			c += 1
	else:
		for i in range(motionVectors.shape[0]):
			for j in range(motionVectors.shape[1]):
				anchor = (int((j + 1 / 2) * tileSize), int((i + 1 / 2) * tileSize))
				tip = (anchor[0] + motionVectors[i, j, 1], anchor[1] + motionVectors[i, j, 0])
				motionDrawing = cv2.arrowedLine(motionDrawing, anchor, tip, **params)
	return motionDrawing

