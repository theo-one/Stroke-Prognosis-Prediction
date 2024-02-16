

def add_sequence_numbers(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        new_lines = []
        current_number = 1
        for line in lines:
            # 检查这一行是否已经有序号
            if line.strip() and not line.lstrip().startswith(tuple(str(i) for i in range(10))):
                new_lines.append(f"{current_number}. {line}")
                current_number += 1
            else:
                new_lines.append(line)
                # 更新当前序号
                if line.strip():
                    try:
                        current_number = int(line.split('.')[0].strip()) + 1
                    except ValueError:
                        # 如果不能从行中提取数字，保持当前序号不变
                        pass

        # 将处理后的文本写回文件
        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(new_lines)

        return "序号添加成功。"
    except FileNotFoundError:
        return "文件未找到，请检查文件路径。"
    except Exception as e:
        return f"发生错误：{e}"

file_path = "./IST_column_names.md"
add_sequence_numbers(file_path)
