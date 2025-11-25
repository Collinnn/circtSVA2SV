#!/usr/bin/env python3
import re
import sys
import os

def transform(text):
  lines = text.splitlines()
  new_lines = []
  ref_map = {}
  for line in lines:
    match = re.match(r"\s*(%\w+) = moore\.read (%\w+) : <l1>", line)
    if match:
      temp, src = match.groups()
      ref_map[temp] = src
      continue

    m = re.match(r"\s*(%\w+) = moore\.to_builtin_bool (%\w+) : l1", line)
    if m:
      temp, src = m.groups()
      orignal = ref_map[src]
      ref_map[temp] = orignal
      continue
        
    # Match ltl.delay lines
    delay_match = re.match(r"(\s*%(\d+) = ltl\.delay )(%\d+), (\d*), (\d+)(.*)", line)
    clock_match = re.match(r"(\s*%[\w\d_]+ = ltl\.clock )(%[\w\d_]+), posedge (%\d+)(.*)", line)
    implication_match = re.match(r"(\s*%(\d+)\s*=\s*ltl\.implication )(%\d+), (%\d+)(.*)", line)
    if delay_match:
      indent, new_id, arg, delay, length, rest = delay_match.groups()
      if arg in ref_map:
        orignal = ref_map[arg]
        split = orignal.split("_")[0]
        line = f"{indent}{split}, {delay}, {length} {rest}"
    
    elif clock_match:
      indent, arg, clk, rest = clock_match.groups()
      if arg in ref_map:
        orignal = ref_map[arg]
        arg = orignal.split("_")[0]
      if clk in ref_map:
        orignal_clk = ref_map[clk]
        split_clk = orignal_clk.split("_")[0]
      line = f"{indent}{arg}, posedge {split_clk}{rest}"
    elif implication_match:
      indent, new_id, arg1, arg2, rest = implication_match.groups()
      if arg1 in ref_map:
        orignal1 = ref_map[arg1]
        arg1 = orignal1.split("_")[0]
      if arg2 in ref_map:
        orignal2 = ref_map[arg2]
        arg2 = orignal2.split("_")[0]
      line = f"{indent}{arg1}, {arg2}{rest}"
    
    new_lines.append(line)

  return "\n".join(new_lines)

def transform_module_signature(text):
    """
    Transforms moore.module into hw.module, replaces input types : !moore.l1 with : i1,
    removes the 'moore.procedure always {' line, and decreases indentation of its inner lines by one level.
    """
    # 1. Replace 'moore.module' with 'hw.module'
    text = text.replace("moore.module", "hw.module")

    # 2. Replace input types in the module signature
    text = re.sub(r"in (%\w+) : !moore\.l1", r"in \1 : i1", text)

    lines = text.splitlines()
    new_lines = []
    skip_line = False
    inside_always = False

    for line in lines:
        # Detect start of moore.procedure always
        if re.match(r"\s*moore\.procedure always\s*{", line):
            inside_always = True
            continue  # skip the always line itself

        # Detect end of block
        if inside_always and re.match(r"\s*}", line):
            inside_always = False
            continue  # skip the closing brace
        
        if "moore." in line:
            continue

        # If inside the block, remove one level of indentation (assume 1-2 spaces)
        if inside_always:
            new_lines.append(re.sub(r"^ {1,2}", "", line))  # remove 1-2 leading spaces
        else:
            new_lines.append(line)

    return "\n".join(new_lines)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 transform.py <input.mlir> <output.mlir>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    if not os.path.exists(input_path):
        print(f"Error: file '{input_path}' does not exist.")
        sys.exit(1)

    with open(input_path, "r") as f:
        input_text = f.read()

    output_text = transform(input_text)
    output_text = transform_module_signature(output_text)

    with open(output_path, "w") as f:
        f.write(output_text)

    print(f"Transformed '{input_path}' → '{output_path}'")
