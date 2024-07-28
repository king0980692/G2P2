#!/bin/bash

huggingface-cli download Leon-Chang/exp --repo-type dataset --local-dir ./tmp/

mkdir -p res/Musical_Instruments/

huggingface-cli download Leon-Chang/g2p2_ckpts --repo-type dataset --local-dir ./res/Musical_Instruments/
