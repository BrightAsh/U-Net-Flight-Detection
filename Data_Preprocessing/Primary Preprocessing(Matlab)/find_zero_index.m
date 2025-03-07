function zero_index = find_zero_index(label_data)
    zero_index = 0;
    for i = 1:length(label_data)
        if label_data(i) == 0
            zero_index = i;
            break;
        end
    end
end
