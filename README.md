
# **Long-running Experiments Monitoring**

This project demonstrates how to monitor and log long-running deep learning experiments using Python, PyTorch, TensorBoard, and MLflow.

## **Setup**

1. **Clone the repository:**

   ```
   git clone https://github.com/revanthchristober/Monitoring-Long-Running-Experiments.git
   cd long_running_experiments
   ```

2. **Install the required packages:**

   ```
   pip install -r requirements.txt
   ```

## **Running Experiments**

You can run different experiments by specifying the `--experiment` argument when executing the `main.py` script.

### Running Experiment 1

```
python scripts/main.py --experiment experiment1
```

### Running Experiment 2

```
python scripts/main.py --experiment experiment2
```

## **Monitoring Experiments**

### Monitoring with TensorBoard

To start TensorBoard and monitor the logs:

```
python scripts/monitor.py
```

Then, open your web browser and go to `http://localhost:6006/` to view the TensorBoard dashboard.

## **Project Structure**

- `data/`: Contains data loading scripts.
- `experiments/`: Contains experiment scripts.
- `logs/`: Contains log files.
- `models/`: Contains model definition and utility scripts.
- `scripts/`: Contains main scripts to run training and monitoring.

## **Maintenance Guide**

1. **Log Maintenance**: Regularly clean up old logs to save disk space. Use automated scripts to archive or delete logs beyond a certain date.
2. **Experiment Monitoring**: Set up alerts for system resource usage (CPU, GPU, memory) to prevent crashes.
3. **Model Versioning**: Use MLflow to track different versions of models, hyperparameters, and datasets.
4. **Documentation**: Keep your README and code comments up-to-date with changes in the experiment setup.

## **License**

This project is licensed under the MIT License - see the LICENSE.md file for details.
