import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_config', '-r',
                        default='demo_config.json',
                        type=str,
                        help='Path of running SOTA configuration')
    args = parser.parse_args()
    with open(args.run_config) as f:
        configs = json.loads(f.read())
        #print(configs)
    for config in configs:
        print(f'Running {config["name"]} on {config["input file"]}...')
        if config["input file"].startswith("isolet_shake"):
            print(f"Skipping {config['name']} on {config['input file']} due to large data size.")
            continue
        if config['name'] == 'STORM':
            from SOTA import STORM
            STORM.main(config)
        elif config['name'] == 'HSTree':
            from SOTA import HSTree
            HSTree.main(config)
        elif config['name'] == 'IForestASD':
            from SOTA import IForestASD
            IForestASD.main(config)
        elif config['name'] == 'LODA':
            from SOTA import LODA
            LODA.main(config)
        elif config['name'] == 'RSHash':
            from SOTA import RSHash
            RSHash.main(config)
        elif config['name'] == 'xStream':
            from SOTA import xStream
            xStream.main(config)
        elif config['name'] == 'RRCF':
            from SOTA import RRCF
            RRCF.main(config)
        elif config['name'] == 'Memstream':
            from SOTA import Memstream
            Memstream.main(config)
        elif config['name'] == 'IDKs':
            from SOTA import IDKs
            IDKs.main(config)
        elif config['name'] == 'INNEs':
            from SOTA import INNEs
            INNEs.main(config)
        elif config['name'] == 'ARCUS':
            pass



if __name__ == '__main__':
    main()