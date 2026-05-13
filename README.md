# Initialization

## CloudLab

> Hardware Information: https://docs.cloudlab.us/hardware.html - CloudLab Clemson, `nvidiagh`
> Experiment Profile Link: https://www.cloudlab.us/p/gpu-coherence/grace-hopper

Instantiate a new experiment, using the profile listed above on one of the `nvidiagh` nodes available through CloudLab's Clemson Datacenter. 

## Start from Scratch

Follow these steps to initialize your CUDA Environment. 

```
sudo DEBIAN_FRONTEND=noninteractive apt purge linux-image-$(uname -r) linux-headers-$(uname -r) linux-modules-$(uname -r) -y
sudo apt update
sudo apt install linux-nvidia-64k-hwe-22.04 -y
sudo reboot now
```
```
sudo apt install linux-headers-$(uname -r)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring*.deb
sudo apt update
sudo apt install cuda-toolkit-12-4 -y
sudo apt install nvidia-kernel-open-550 cuda-drivers-550 -y
sudo mkdir /lib/systemd/system/nvidia-persistenced.service.d
sudo dd status=none of=/lib/systemd/system/nvidia-persistenced.service.d/override.conf << EOF
[Service]
ExecStart=/usr/bin/nvidia-persistenced --persistence-mode --verbose
[Install]
WantedBy=multi-user.target 
EOF
sudo su
echo "kernel.numa_balancing = 0" >> /etc/sysctl.conf
sudo reboot now
sudo apt install numactl nvidia-cuda-toolkit build-essential -y
```

# Clone this Repository

```
git clone https://github.com/icsa-caps/ismm26-artifacts.git
git submodule update --init --recursive
```

# Running Grace Litmus Tests

The `grace-litmus` directory hosts the `herdtools7` suite, out of which we require diyone7 and litmus7. 

# Running Hopper Litmus Tests

```
cd hopper-litmus
./tune-specific.sh tune-mp.txt > output.log 
python3 parse_stats.py output.log
# produces output_parsed.csv
```

`tune-specific.sh` runs all the MP variants one after the other in an infinite loop. The litmus testing must be terminated manually using `Ctrl+C`. Even without terminating the litmus testing, you can run the `parse_stats.py` to parse whatever data has been collected thus far. 

> Observing weak behaviours is probabilistic. If you don't observe weak behaviours in the test variants where it is expected, run the testing suite for longer.
> For some variants, it took several days for the weak behaviours to reveal themselves. 

# Running Grace-Hopper Litmus Tests

```
cd gh-litmus
make -C stress_malloc
make -C stress_malloc/reverse
python3 run-all.py
```

Similar to the _Hopper Litmus Tests_, the Grace-Hopper litmus tests also run in an infinite loop and need to be terminated manually using `Ctrl+C`. 

# Running Value Propagation Tests

