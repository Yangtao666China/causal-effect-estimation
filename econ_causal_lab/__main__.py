import argparse
import json


def main():
    from .experiment import benchmark_run,custom_run
    parser = argparse.ArgumentParser(description="Reproducible causal ML: estimates, overlap and sensitivity.")
    commands = parser.add_subparsers(dest="command",required=True)
    benchmark = commands.add_parser("benchmark",help="NSW/CPS study plus known-effect simulations")
    benchmark.add_argument("--cache",default="data/raw")
    benchmark.add_argument("--offline",action="store_true")
    benchmark.add_argument("--repetitions",type=int,default=100,help="Monte Carlo repetitions per scenario; 0 skips simulations")
    benchmark.add_argument("--simulation-n",type=int,default=1000)
    custom = commands.add_parser("analyze",help="Analyze a numeric CSV; identification assumptions are your responsibility")
    custom.add_argument("csv")
    custom.add_argument("--outcome",required=True)
    custom.add_argument("--treatment",required=True)
    custom.add_argument("--features",required=True,help="comma-separated pre-treatment numeric covariates")
    for sub in (benchmark,custom):
        sub.add_argument("--output",default="outputs/report")
        sub.add_argument("--folds",type=int,default=5)
        sub.add_argument("--seeds",default="42,7,2026")
        sub.add_argument("--clip",type=float,default=.01)
    args = parser.parse_args()
    try:
        seeds = tuple(int(s.strip()) for s in args.seeds.split(","))
        shared = dict(output=args.output,folds=args.folds,seeds=seeds,clip=args.clip)
        if args.command == "benchmark":
            if args.repetitions < 0 or args.repetitions == 1:
                raise ValueError("repetitions must be 0 or at least 2")
            report = benchmark_run(cache=args.cache,offline=args.offline,repetitions=args.repetitions,
                                   simulation_n=args.simulation_n,**shared)
        else:
            features = [s.strip() for s in args.features.split(",") if s.strip()]
            report = custom_run(args.csv,features,args.outcome,args.treatment,**shared)
    except (ValueError,OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({"output":args.output,"results":report["results"]},ensure_ascii=True,indent=2))


if __name__ == "__main__":
    main()
