%% Sets bad events manually
function D = set_bad_events(D, BadEpochs, modality, remove)
    % save bad epochs using method meeg/events
    BadEvents = struct([]);
    for j = 1:numel(BadEpochs)
        if numel(BadEpochs{j}) == 2
            BadEvents(j).type = 'artefact_OSL';
            BadEvents(j).value = modality;
            BadEvents(j).time = BadEpochs{j}(1);

            %+ 2/D.fsample; % Need to account for SPM12's weird rounding
            BadEvents(j).duration = diff(BadEpochs{j});
            BadEvents(j).offset = 0;
        end
    end
    
    % load events
    ev = D.events;
        
    % remove previous bad epoch events for this
    if isfield(ev,'type') && remove
        to_remove = false(size(ev));
        for j = 1:length(ev)
            if strcmp(ev(j).type, 'artefact_OSL') &&...
                    ismember(ev(j).value, {modality})
                to_remove(j) = 1;
            end
        end
        ev(to_remove) = [];
    end
    
    if ~isempty(BadEvents)
        ev = [ev(:); BadEvents(:)];
    end
    
    % save new events with previous
    D = events(D,1,ev);
end