from modelos import *
ensemble("ENS HGB S2 + HGB todas-TE", ["HGB S2 l2=5","HGB todas-TE"])
ensemble("ENS HGB S2 + HGB todas l2=5", ["HGB S2 l2=5","HGB todas l2=5"])
ensemble("ENS HGB S2 + HGB todas-TE + RF S", ["HGB S2 l2=5","HGB todas-TE","RF S sem cw"])
ensemble("ENS HGB S2 + HGB orig84+H+outras", ["HGB S2 l2=5","HGB orig84+H+outras"])
ensemble("ENS HGB S2 + HGB todas-TE + HGB todas mono", ["HGB S2 l2=5","HGB todas-TE","HGB todas l2=5 mono"])
