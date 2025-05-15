#!/usr/bin/env python3

def fix_app_py():
    with open('app.py', 'r') as file:
        lines = file.readlines()
    
    # Find the problem area
    for i in range(len(lines)):
        if 'topic_result["summary"] = summary' in lines[i] and i < len(lines) - 3:
            if 'else:' in lines[i+1]:
                # This is where the problem is
                problem_area = i
                break
    
    # Fix the syntax error
    fixed_lines = []
    for i, line in enumerate(lines):
        if i == problem_area:
            # Add correct indentation and structure
            fixed_lines.append('                                topic_result["summary"] = summary\n')
            fixed_lines.append('                            else:\n')
            fixed_lines.append('                                topic_result["summary"] = f"Failed to summarize topic: {topic_name}"\n')
            fixed_lines.append('                    else:\n')
            fixed_lines.append('                        topic_result["summary"] = f"No sentences found for topic: {topic_name}"\n')
            # Skip the next few lines which are incorrectly formatted
            i += 4
        elif i > problem_area and i <= problem_area + 4:
            # We've already added these lines in corrected form
            continue
        else:
            fixed_lines.append(line)
    
    # Write the fixed content back
    with open('app.py', 'w') as file:
        file.writelines(fixed_lines)
    
    print("Fixed syntax error in app.py")

if __name__ == '__main__':
    fix_app_py() 