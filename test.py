def detect_sign_change(previous, current):
    if previous >= 0 and current < 0:
        return 'positive_to_negative'
    elif previous < 0 and current >= 0:
        return 'negative_to_positive'
    else:
        return 'no_change'


# Example usage
previous_number = 3
current_number = -5

result = detect_sign_change(previous_number, current_number)
print(f"Sign change: {result}")
