ch_path = 'rich_data/opm_lukas/fixed/20221019_085012_channels.tsv';

ch_table = readtable(ch_path,'FileType','text','Delimiter','tab');
   
ch_table.isx = endsWith(ch_table.name,'[X]');
ch_table.isy = endsWith(ch_table.name,'[Y]');
ch_table.isz = endsWith(ch_table.name,'[Z]');
ch_table.slot_no = zeros(height(ch_table),1);
% sanity check
if sum(sum(ch_table{:,11:13},2)) ~= height(ch_table)
    error('Channel orientation [x,y,z] labels might be wrong!')
end

load('rich_data/opm_lukas/fixed/20221019_085012_sensor_order.mat','T')
for sl_i = 1:height(T)
    if ~isempty(T{sl_i,1}{1})
        ch_table.slot_no(startsWith(ch_table.name,T{sl_i,1}{1}(1:2))) = sl_i;
    end
end