import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Define the data
data = {
    "Size": [1000000, 100000000],
    "op1_speedup": [1.334802117, 1.251852714],
    "op2_speedup": [0.03170678395, 0.617627468],
    "op3_speedup": [1.404938098, 3.178227403]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Plotting the bar graph
fig, ax = plt.subplots(figsize=(8, 6))

# Bar width
bar_width = 0.2
index = np.arange(len(df))

# Create bars for each optimization
bar2 = ax.bar(index + bar_width, df['op1_speedup'], bar_width, label='op1', color='green')
bar3 = ax.bar(index + 2 * bar_width, df['op2_speedup'], bar_width, label='op2', color='red')
bar4 = ax.bar(index + 3 * bar_width, df['op3_speedup'], bar_width, label='op3', color='purple')

# Set the labels and title
ax.set_xlabel('Size')
ax.set_ylabel('Speedup')
ax.set_title('Speedup for Different Optimizations')
ax.set_xticks(index + 1.5 * bar_width)  # Adjusted for better positioning
ax.set_xticklabels(df['Size'])
ax.legend()

# Save the plot
plt.tight_layout()
plt.savefig('cpu_speedup.png', format='png')

# Close the plot to free memory
plt.close()
