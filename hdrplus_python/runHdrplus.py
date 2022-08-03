"""Script to run the hdrplus implementation on a single burst.
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

This file implements an algorithm possibly linked to the patent US9077913B2.
This file is made available for the exclusive aim of serving as scientific tool
to verify the soundness and completeness of the algorithm description.
Compilation, execution and redistribution of this file may violate patents rights in certain countries.
The situation being different for every country and changing over time,
it is your responsibility to determine which patent rights restrictions apply to you
before you compile, use, modify, or redistribute this file.
A patent lawyer is qualified to make this determination.
If and only if they don't conflict with any patent terms,
you can benefit from the following license terms attached to this file.
"""

# imports
import os
import time
import argparse
# custom package imports
from package.algorithm.hdrplus import hdrplusPipeline
from package.algorithm.genericUtils import getTime
from package.algorithm.params import getParams


def getParsers():
    '''Command line argument parsing.'''
    # Initialize the parser
    parser = argparse.ArgumentParser(add_help=True)

    # Input/output arguments
    parser.add_argument(
        '-i',
        '--input',
        required=True,
        type=str,
        help='Input folder.'
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default='',
        help='Output directory. If not provided, will use the input folder.'
    )
    parser.add_argument(
        '-m',
        '--mode',
        type=str,
        default='full',  # 'full' 'align' 'merge' 'finish'
        help='Mode to which you want to run the program. If not specified, will output a single fully processed image from a raw burst.'
    )
    parser.add_argument(
        '-r',
        '--reference',
        type=int,
        default=0,
        help="Reference image index (int between 0 and N-1 / -1 for index stored in reference.txt file). Only applies to 'full' or 'align' algorithm modes"
    )
    parser.add_argument(
        '-tf',
        '--temporalfactor',
        type=float,
        default=75.,
        help="Temporal denoising factor (float > 0, default=75). Only applies to 'full' or 'merge' algorithm modes"
    )
    parser.add_argument(
        '-sf',
        '--spatialfactor',
        type=float,
        default=0.1,
        help="Spatial denoising factor (float >= 0., default=0.1). Only applies to 'full' or 'merge' algorithm modes"
    )
    parser.add_argument(
        '-ltm',
        '--ltmgain',
        type=int,
        default=-1,
        help="Local tone mapping gain (int > 0 / -1 for automatic computation). Only applies to 'full' or 'finish' algorithm modes"
    )
    parser.add_argument(
        '-gtm',
        '--gtmcontrast',
        type=float,
        default=0.075,
        help="Global tone mapping contrast enhancement ratio (float between 0. and 1., default=0.075). Only applies to 'full' or 'finish' algorithm modes"
    )
    parser.add_argument(
        '-v',
        '--verbose',
        type=int,
        choices=(0, 1, 2, 3, 4, 5),
        default=0,
        help='Verbosity level'
    )

    return parser.parse_args()


def getOptions(args):
    '''For convenience. Set arguments options in dictionary for later use.'''
    return {
        'outputFolder': args.output if args.output != '' else args.input,
        'inputFolder': args.input,
        'mode': args.mode,
        'referenceIndex': args.reference,
        'temporalFactor': args.temporalfactor,
        'spatialFactor': args.spatialfactor,
        'ltmGain': args.ltmgain,
        'gtmContrast': args.gtmcontrast,
        'verbose': args.verbose
    }


if __name__ == "__main__":

    # Get the arguments
    options = getOptions(getParsers())
    # Get the parameters that correspond to the selected mode
    params = getParams(options['mode'])

    # Create output folder if needed
    if not os.path.isdir(options['outputFolder']):
        os.makedirs(options['outputFolder'])

    if not os.path.isdir(options['outputFolder'] + '/alignments'):
        os.makedirs(options['outputFolder'] + '/alignments')

        # For convenience
    currentTime, verbose = time.time(), options['verbose'] > 0

    burstPath = options['inputFolder']
    # Process the burst
    if verbose:
        print("=" * (20 + len(burstPath)))
        print("Processing burst: {}".format(burstPath))
        print("=" * (20 + len(burstPath)))
        # Run the pipeline on the burst
    hdrplusPipeline(burstPath, params, options)
    if verbose:
        rentTime = getTime(currentTime, ' - Burst processed')
