import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Define the data
data = {
    "Size": [1000000, 100000000],
    "op1_speedup": [3811.001189, 63329.14041],
    "op2_speedup": [3253.94069, 56538.53067],
    "op3_speedup": [826.9153523, 31253.77162]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Plotting the bar graph
fig, ax = plt.subplots(figsize=(8, 6))

# Bar width
bar_width = 0.2
index = np.arange(len(df))

# Create bars for each optimization
bar1 = ax.bar(index, df['op1_speedup'], bar_width, label='op1', color='blue')
bar2 = ax.bar(index + bar_width, df['op2_speedup'], bar_width, label='op2', color='green')
bar3 = ax.bar(index + 2 * bar_width, df['op3_speedup'], bar_width, label='op3', color='red')

# Set the labels and title
ax.set_xlabel('Size')
ax.set_ylabel('Speedup')
ax.set_title('Speedup for Different Optimizations')
ax.set_xticks(index + bar_width)
ax.set_xticklabels(df['Size'])
ax.legend()

# Save the plot
plt.tight_layout()
plt.savefig('speedup_for_optimizations.png', format='png')

# Close the plot to free memory
plt.close()
