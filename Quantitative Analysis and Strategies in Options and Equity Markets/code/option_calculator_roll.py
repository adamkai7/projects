import optmodel as opt
from numpy import *

import sys


from volta.utils import expand_path, time_to_slice, parse_date_range
from volta.cmdparser import code_relative_path, CommandParser
from niven.data import load_catalog, load_aligned_series, StockDataset
from sklearn.neighbors import NearestNeighbors


'''
Created by Adam Kainikara

How to run it:
python option_calculator_roll.py --s0 100 --k0 100 --dte 90 --iv 0.25 --rwsize 30 --k1 100

This file serves to act as a calculator that answers the question, “At what future underlying price will a new option position 
(with shorter maturity and possibly a different strike) have the same value as my current option?” This is essentially the mechanics 
of an option roll (closing one option and opening another, usually further out in time or at a different strike) or trying to leg 
into a position by purchasing an additional position.

The inputted parameters can be set from the command line (example above)
The code is designed to be run from the command line, with the user able to input their current option's initial underlying value 
(s0), initial strike (k0), strike of the target (rolled) option, days to expiry of the current option (dte0), number of days rolled 
forward i.e., how much closer to expiry the new option is (rwsize), and current implied volatility (iv). 
'''
def option_value(s, k, dte, iv, rf, q):
    t = dte / 365
    return opt.gbs_call(s, k, t, iv, rf, q).value


def parse_args():
    cmd = CommandParser()
    cmd.arg('--s0', type=float, required=True)
    cmd.arg('--k0', type=float, required=True)
    cmd.arg('--k1', type=float, required=True)
    cmd.arg('--dte0', type=float, required=True)
    cmd.arg('--rwsize', type=float, required=True)
    cmd.arg('--iv', type=float, default=0.2068, help='Implied Volatility')
    cmd.arg('--rf', type=float, default=0.0425, help='Risk-Free Rate')
    cmd.arg('--q', type=float, default=0.0043, help='Dividend Yield')

    cmd.arg('--pct_range', type=float, default=0.15, help='Percentage Range around s0')
    cmd.arg('--num_points', type=int, default=100, help='Number of points in the underlying range')

    args = cmd.eval()
    return args


def find_required_underlying_and_range(s0, k0, k1, dte0, rwsize, iv=0.2068, rf=0.0425, q=0.0043, pct_range=0.15, num_points=100):
    v0 = option_value(s0, k0, dte0, iv, rf, q)

    s_low = s0
    s_high = s0 + s0 * v0

    tolerance = 1e-5  # stop when v1 is within this distance to v0

    while True:
        s_mid = (s_low + s_high) / 2
        v1 = option_value(s_mid, k1, dte0 - rwsize, iv, rf, q)

        if abs(v1 - v0) <= tolerance:
            break

        if v1 > v0:
            s_high = s_mid
        else:
            s_low = s_mid

    s_required = s_mid
    v1_required = v1

    s_min = s0 * (1 - pct_range)
    s_max = s0 * (1 + pct_range)
    s_range_v = linspace(s_min, s_max, num_points)
    v1_v = [option_value(s, k1, dte0 - rwsize, iv, rf, q) for s in s_range_v]

    return {
            'initial_value': v0,
            'required_underlying': s_required,
            'required_value': v1_required,
            'underlying_range': s_range_v,
            'option_values': v1_v}


def compute_spread(s_required, k0, k1, dte0, rwsize, iv, rf, q):
    v0_s1 = option_value(s_required, k0, dte0, iv, rf, q)
    v1_required = option_value(s_required, k1, dte0 - rwsize, iv, rf, q)
    spread = v1_required - v0_s1
    return {
            'v0_at_s1': v0_s1,
            'v1_at_s1': v1_required,
            'spread': spread}


def main():
        # s0, k0, k1 = 555, 556, 570
        # dte0, rwsize = 60, 20
        # s0, k0, k1, dte0, rwsize = map(float, sys.argv[1:6])
    args = parse_args()

    result_d = find_required_underlying_and_range(args.s0, args.k0, args.k1, args.dte0, args.rwsize, iv=args.iv, rf=args.rf, q=args.q,
            pct_range=args.pct_range, num_points=args.num_points)
        # result_d = find_required_underlying_and_range(s0, k0, k1, dte0, rwsize)

    #    result_d = find_required_underlying_and_range(s0, k0, k1, dte0, rwsize)

    print(f"\nInitial Value (V0): {result_d['initial_value']:.4f}")
    print(f"Required Underlying (S1): {result_d['required_underlying']:.4f} "
            f"Final Value (V1): {result_d['required_value']:.4f}")

    spread_result_d = compute_spread(result_d['required_underlying'], args.k0, args.k1, args.dte0, args.rwsize, iv=args.iv, rf=args.rf, 
    q=args.q)

    print(f"\nV0 at S1 (K0 strike): {spread_result_d['v0_at_s1']:.4f}")
    print(f"V1 at S1 (K1 strike): {spread_result_d['v1_at_s1']:.4f}")
    print(f"Spread (V1 - V0): {spread_result_d['spread']:.4f}")
    #print(type(result_d))
    raise SystemExit


if __name__ == '__main__':
    main()



'''    
How to run it:
python option_calculator_roll.py --s0 100 --k0 100 --dte 90 --iv 0.25 --rwsize 30 --k1 100

This file serves to act as a calculator that answers the question, “At what future underlying price will a new option position 
(with shorter maturity and possibly a different strike) have the same value as my current option?” This is essentially the mechanics 
of an option roll (closing one option and opening another, usually further out in time or at a different strike) or trying to leg 
into a position by purchasing an additional position.

The inputted parameters can be set from the command line (example above)
The code is designed to be run from the command line, with the user able to input their current option's initial underlying value 
(s0), initial strike (k0), strike of the target (rolled) option, days to expiry of the current option (dte0), number of days rolled 
forward i.e., how much closer to expiry the new option is (rwsize), and current implied volatility (iv). 

'''