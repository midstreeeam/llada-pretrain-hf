import re
import matplotlib.pyplot as plt
import os
import numpy as np

def strip_ansi(text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def parse_log(file_path):
    steps = []
    train_losses = []
    val_losses = []
    val_steps = []
    step_to_tokens = {}

    # Regex to match the metrics line
    metrics_pattern = re.compile(r"\[metrics\].*step\s+(\d+)\s+(.*)")
    
    with open(file_path, 'r') as f:
        for line in f:
            clean_line = strip_ansi(line)
            match = metrics_pattern.search(clean_line)
            if match:
                step = int(match.group(1))
                rest = match.group(2)
                
                # Extract tokens seen
                token_match = re.search(r"num_input_tokens_seen=([\d\.]+)", rest)
                if token_match:
                    tokens = float(token_match.group(1))
                    step_to_tokens[step] = tokens

                # Check for train loss
                loss_match = re.search(r"loss=([\d\.]+)", rest)
                if loss_match:
                    if "eval_loss=" not in rest:
                        steps.append(step)
                        train_losses.append(float(loss_match.group(1)))
                
                # Check for eval loss
                eval_loss_match = re.search(r"eval_loss=([\d\.]+)", rest)
                if eval_loss_match:
                    val_steps.append(step)
                    val_losses.append(float(eval_loss_match.group(1)))

    return steps, train_losses, val_steps, val_losses, step_to_tokens

def plot_losses(steps, train_losses, val_steps, val_losses, step_to_tokens, output_file="training_plot.png"):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot losses
    if steps and train_losses:
        ax1.plot(steps, train_losses, label='Train Loss', alpha=0.7)
        
    if val_steps and val_losses:
        ax1.plot(val_steps, val_losses, label='Validation Loss', color='orange', marker='o', linestyle='--')

    ax1.set_xlabel('Steps (Log Scale)')
    ax1.set_ylabel('Loss')
    ax1.set_xscale('log')
    ax1.set_title('Training and Validation Loss')
    ax1.legend(loc='upper right')
    ax1.grid(True, which="both", ls="-", alpha=0.5)
    
    # Use proper logarithmic tick spacing
    from matplotlib.ticker import LogLocator, LogFormatterSciNotation, ScalarFormatter
    
    # Set major ticks at powers of 10 and intermediate values (including more granular ones)
    ax1.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), numticks=30))
    
    # Use scalar formatter to show actual numbers instead of scientific notation
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax1.xaxis.set_major_formatter(formatter)
    
    # Set x-axis limits to start from a smaller value if we have data
    if steps:
        min_step = min(steps)
        max_step = max(steps)
        # Start from 10000 or the nearest power of 10 below min_step
        start_step = 10 ** (np.floor(np.log10(min_step)))
        ax1.set_xlim(left=start_step, right=max_step * 1.1)
    
    # Rotate labels for better readability
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Secondary x-axis for tokens
    if step_to_tokens:
        # Calculate average tokens per step to define the transformation
        # We use the max step to get a stable ratio
        max_step = max(step_to_tokens.keys())
        max_tokens = step_to_tokens[max_step]
        tokens_per_step = max_tokens / max_step
        
        def step_to_token_func(x):
            return x * tokens_per_step

        def token_to_step_func(x):
            return x / tokens_per_step

        secax = ax1.secondary_xaxis('top', functions=(step_to_token_func, token_to_step_func))
        secax.set_xlabel('Tokens Seen')

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    # Use absolute path as per previous context, or relative if run from root
    log_file = "/home/midstream/workspace/llada-pretrain-hf/output/llada_135m_ts_v/logs/train_20251126_112211.log"
    
    if not os.path.exists(log_file):
        print(f"Error: Log file not found at {log_file}")
    else:
        steps, train_losses, val_steps, val_losses, step_to_tokens = parse_log(log_file)
        
        print(f"Found {len(steps)} training data points.")
        print(f"Found {len(val_steps)} validation data points.")
        
        if not steps and not val_steps:
            print("No metrics found in log file.")
        else:
            plot_losses(steps, train_losses, val_steps, val_losses, step_to_tokens)
